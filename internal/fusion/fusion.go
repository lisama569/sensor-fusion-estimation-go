// Package fusion wires alignment, the EKF and per-channel fault isolation
// into one estimator that accepts measurements from multiple goroutines.
//
// Concurrency model: Submit takes a short mutex and only enqueues; all
// state-mutating filter work happens in Pump/Run/Flush under the same
// mutex, processing measurements in the alignment buffer's deterministic
// order (timestamp, sensorID, arrivalSeq). Therefore the final state and
// the statistics are independent of goroutine scheduling.
package fusion

import (
	"math"
	"sort"
	"sync"
	"time"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/alignment"
	"github.com/lisama569/sensor-fusion-estimation-go/internal/filter"
	"github.com/lisama569/sensor-fusion-estimation-go/internal/models"
	"github.com/lisama569/sensor-fusion-estimation-go/internal/reject"
	"gonum.org/v1/gonum/mat"
)

// ChannelConfig registers one sensor channel.
type ChannelConfig struct {
	ID    string
	Model models.MeasureModel
	// DivergeNIS is the per-update NIS gate; <=0 means use the global gate.
	DivergeNIS float64
	// DivergeCount isolates the channel after this many consecutive
	// gated/invalid updates.
	DivergeCount int
	// RecoverCount reintegrates the channel after this many consecutive
	// healthy trial updates while isolated.
	RecoverCount int
}

// ChannelStatus is an immutable snapshot of channel health.
type ChannelStatus struct {
	ID          string
	Isolated    bool
	Accepted    int64
	Rejected    int64
	BadStreak   int
	GoodStreak  int
	IsolatedAt  float64
	RecoveredAt float64
}

// Stats are estimator-wide counters.
type Stats struct {
	Accepted   int64
	Rejected   int64
	Duplicates int64
	Stale      int64
	BadTime    int64
	NonFinite  int64
	Degenerate int64
	Diverging  int64
	Isolations int64
	Recoveries int64
	Updates    int64
}

// Config configures the Fuser.
type Config struct {
	Align  alignment.Config
	Filter filter.Config
	Tick   float64 // clock advance per Pump step
	StartT float64
	EndT   float64
}

// Event is one logged decision, suitable for test assertions/traces.
type Event struct {
	Wall      time.Time
	Timestamp float64
	SensorID  string
	Kind      string // accepted|rejected|isolated|recovered|predicted|queued
	Reason    string
	NIS       float64
}

type channelState struct {
	cfg         ChannelConfig
	isolated    bool
	accepted    int64
	rejected    int64
	badStreak   int
	goodStreak  int
	isolatedAt  float64
	recoveredAt float64
	// pendingFaults are timestamps of non-finite submissions whose fault
	// has not yet been reached by the estimator clock. They take effect
	// (count toward isolation) only once the watermark passes them, so a
	// future fault cannot retroactively isolate earlier measurements.
	pendingFaults []float64
}

// Fuser is safe for concurrent Submit calls.
type Fuser struct {
	mu       sync.Mutex
	cfg      Config
	efilter  *filter.Filter
	buf      *alignment.Buffer
	chans    map[string]*channelState
	stats    Stats
	log      []Event
	clock    float64
	initX    *mat.VecDense
	initP    *mat.SymDense
	initedAt float64
}

// New builds a Fuser around the given state model and channels.
func New(sm *models.StateModel, channels []ChannelConfig, cfg Config) (*Fuser, error) {
	if len(channels) == 0 {
		return nil, reject.New(reject.ReasonDimensionMismatch, "at least one channel required")
	}
	if cfg.Tick <= 0 {
		return nil, reject.New(reject.ReasonDimensionMismatch, "tick must be positive, got %v", cfg.Tick)
	}
	fu := &Fuser{
		cfg:     cfg,
		efilter: filter.New(sm, cfg.Filter),
		buf:     alignment.NewBuffer(cfg.Align),
		chans:   make(map[string]*channelState),
		clock:   cfg.StartT,
	}
	seen := make(map[string]struct{})
	for _, ch := range channels {
		if ch.ID == "" {
			return nil, reject.New(reject.ReasonDimensionMismatch, "channel id must be non-empty")
		}
		if _, dup := seen[ch.ID]; dup {
			return nil, reject.New(reject.ReasonDimensionMismatch, "duplicate channel id %q", ch.ID)
		}
		seen[ch.ID] = struct{}{}
		if ch.Model == nil {
			return nil, reject.New(reject.ReasonDimensionMismatch, "channel %q has nil model", ch.ID)
		}
		if ch.DivergeCount <= 0 {
			ch.DivergeCount = 3
		}
		if ch.RecoverCount <= 0 {
			ch.RecoverCount = 2
		}
		fu.chans[ch.ID] = &channelState{cfg: ch}
	}
	return fu, nil
}

// Init seeds the initial estimate.
func (fu *Fuser) Init(x *mat.VecDense, P *mat.SymDense) error {
	fu.mu.Lock()
	defer fu.mu.Unlock()
	if err := fu.efilter.Init(fu.cfg.StartT, x, P); err != nil {
		return err
	}
	fu.initX = mat.VecDenseCopyOf(x)
	fu.initP = cloneSym(P)
	fu.initedAt = fu.cfg.StartT
	return nil
}

// Submit validates and enqueues one measurement. Safe for concurrent use.
// Every rejection returns a distinguishable reason and is counted and
// logged; nothing is dropped silently.
func (fu *Fuser) Submit(mm models.Measurement) error {
	fu.mu.Lock()
	defer fu.mu.Unlock()

	wall := time.Now()
	cs, known := fu.chans[mm.SensorID]
	if !known {
		err := reject.New(reject.ReasonDimensionMismatch, "unknown sensor channel %q", mm.SensorID)
		fu.recordLocked(Event{Wall: wall, Timestamp: mm.Timestamp, SensorID: mm.SensorID,
			Kind: "rejected", Reason: string(err.Reason)})
		return err
	}

	// Static validation (timestamps and values) happens in the buffer,
	// except we also want non-finite values reflected in channel stats.
	if mm.Z != nil {
		for i := 0; i < mm.Z.Len(); i++ {
			v := mm.Z.AtVec(i)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				cs.rejected++
				fu.stats.Rejected++
				fu.stats.NonFinite++
				// Defer the isolation decision to the fault's timestamp.
				cs.pendingFaults = append(cs.pendingFaults, mm.Timestamp)
				sort.Float64s(cs.pendingFaults)
				err := reject.New(reject.ReasonNonFiniteValue,
					"sensor %q at t=%v z[%d]=%v", mm.SensorID, mm.Timestamp, i, v)
				fu.recordLocked(Event{Wall: wall, Timestamp: mm.Timestamp, SensorID: mm.SensorID,
					Kind: "rejected", Reason: string(err.Reason)})
				return err
			}
		}
	}

	if err := fu.buf.Accept(mm); err != nil {
		cs.rejected++
		fu.stats.Rejected++
		switch reject.Of(err) {
		case reject.ReasonDuplicateTimestamp:
			fu.stats.Duplicates++
		case reject.ReasonStaleTimestamp:
			fu.stats.Stale++
		case reject.ReasonBadTimestamp:
			fu.stats.BadTime++
		}
		fu.recordLocked(Event{Wall: wall, Timestamp: mm.Timestamp, SensorID: mm.SensorID,
			Kind: "rejected", Reason: err.Error()})
		return err
	}

	fu.recordLocked(Event{Wall: wall, Timestamp: mm.Timestamp, SensorID: mm.SensorID,
		Kind: "queued"})
	return nil
}

// Pump advances the clock by one tick and processes all released
// measurements in deterministic order.
func (fu *Fuser) Pump() ([]Event, error) {
	fu.mu.Lock()
	defer fu.mu.Unlock()
	return fu.advanceLocked(fu.clock + fu.cfg.Tick)
}

// AdvanceTo moves the estimator clock to t (monotonic) and processes all
// measurements released by that watermark. It lets a caller drive the
// clock by data-arrival watermarks rather than by fixed ticks.
func (fu *Fuser) AdvanceTo(t float64) ([]Event, error) {
	fu.mu.Lock()
	defer fu.mu.Unlock()
	return fu.advanceLocked(t)
}

// Run pumps until the configured end time and returns all logged events.
func (fu *Fuser) Run() ([]Event, error) {
	fu.mu.Lock()
	defer fu.mu.Unlock()
	if !fu.efilter.Initialized() {
		return nil, reject.New(reject.ReasonNotInitialized, "Run before Init")
	}
	var firstErr error
	for fu.clock < fu.cfg.EndT-1e-12 {
		next := fu.clock + fu.cfg.Tick
		if next > fu.cfg.EndT {
			next = fu.cfg.EndT
		}
		if _, err := fu.advanceLocked(next); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return fu.snapshotEventsLocked(), firstErr
}

// Flush releases all queued data and processes it.
func (fu *Fuser) Flush() ([]Event, error) {
	fu.mu.Lock()
	defer fu.mu.Unlock()
	ms, err := fu.buf.Flush()
	if err != nil {
		return fu.snapshotEventsLocked(), err
	}
	target := fu.clock
	for _, mm := range ms {
		if mm.Timestamp > target {
			target = mm.Timestamp
		}
	}
	// Predict without changing queued set: flush already released all.
	if err := fu.predictLocked(target); err != nil {
		return fu.snapshotEventsLocked(), err
	}
	fu.applyDueFaultsLocked(target)
	for _, mm := range ms {
		fu.applyLocked(mm)
	}
	return fu.snapshotEventsLocked(), nil
}

// DrainAndAdvance is the Pump body on the caller's target time.
func (fu *Fuser) advanceLocked(target float64) ([]Event, error) {
	if !fu.efilter.Initialized() {
		return nil, reject.New(reject.ReasonNotInitialized, "pump before Init")
	}
	if err := fu.predictLocked(target); err != nil {
		return nil, err
	}
	fu.applyDueFaultsLocked(target)
	released := fu.buf.AdvanceTo(target)
	for _, mm := range released {
		fu.applyLocked(mm)
	}
	return fu.snapshotEventsLocked(), nil
}

func (fu *Fuser) predictLocked(target float64) error {
	if err := fu.efilter.PredictTo(target); err != nil {
		return err
	}
	fu.clock = target
	fu.recordLocked(Event{Timestamp: target, Kind: "predicted"})
	return nil
}

// applyLocked processes one released measurement: isolation gate, trial
// update while isolated, NIS-based fault detection and auto-recovery.
func (fu *Fuser) applyLocked(mm models.Measurement) {
	wall := time.Now()
	cs := fu.chans[mm.SensorID]
	md := cs.cfg.Model

	// While isolated the channel is excluded from the global estimate; we
	// perform a *trial* update on a scratch copy of the filter state so a
	// faulty channel can never pollute the global estimate, and use the
	// trial outcome to decide recovery.
	if cs.isolated {
		trial := fu.scratchFilterLocked()
		out, err := trial.Update(mm.SensorID, mm, md)
		cs.rejected++
		fu.stats.Rejected++
		healthy := err == nil && out.Accepted && !fu.gateTrippedLocked(cs, out.NIS)
		if healthy {
			cs.goodStreak++
		} else {
			cs.goodStreak = 0
		}
		reason := string(reject.ReasonChannelIsolated)
		if err != nil {
			reason = err.Error()
		}
		fu.recordLocked(Event{Wall: wall, Timestamp: mm.Timestamp, SensorID: mm.SensorID,
			Kind: "rejected", Reason: reason, NIS: out.NIS})
		if healthy && cs.goodStreak >= cs.cfg.RecoverCount {
			cs.isolated = false
			cs.badStreak = 0
			cs.goodStreak = 0
			cs.recoveredAt = mm.Timestamp
			fu.stats.Recoveries++
			// Commit the healthy trial state to the live filter.
			fu.efilter = trial
			cs.accepted++
			fu.stats.Accepted++
			fu.recordLocked(Event{Wall: wall, Timestamp: mm.Timestamp, SensorID: mm.SensorID,
				Kind: "recovered", NIS: out.NIS})
		}
		return
	}

	// Every live update is pre-tried on a scratch copy so an invalid or
	// diverging observation can never mutate the global estimate.
	trial := fu.scratchFilterLocked()
	out, err := trial.Update(mm.SensorID, mm, md)
	fu.stats.Updates++
	if err != nil {
		cs.rejected++
		cs.badStreak++
		cs.goodStreak = 0
		fu.stats.Rejected++
		switch reject.Of(err) {
		case reject.ReasonNonFiniteValue, reject.ReasonNonFiniteInnovation:
			fu.stats.NonFinite++
		case reject.ReasonDegenerateUpdate:
			fu.stats.Degenerate++
		}
		fu.recordLocked(Event{Wall: wall, Timestamp: mm.Timestamp, SensorID: mm.SensorID,
			Kind: "rejected", Reason: err.Error(), NIS: out.NIS})
		fu.maybeIsolateLocked(cs, mm.Timestamp, wall)
		return
	}
	if fu.gateTrippedLocked(cs, out.NIS) {
		// Gate trip: the trial result is discarded. One trip is a legal
		// large observation and does not isolate; DivergeCount consecutive
		// trips do.
		cs.rejected++
		cs.badStreak++
		cs.goodStreak = 0
		fu.stats.Rejected++
		fu.stats.Diverging++
		fu.recordLocked(Event{Wall: wall, Timestamp: mm.Timestamp, SensorID: mm.SensorID,
			Kind: "rejected", Reason: string(reject.ReasonDiverging), NIS: out.NIS})
		fu.maybeIsolateLocked(cs, mm.Timestamp, wall)
		return
	}

	// Healthy: commit the exact trial result to the live filter by simply
	// replacing it with the trial.
	fu.efilter = trial
	cs.accepted++
	cs.badStreak = 0
	cs.goodStreak = 0
	fu.stats.Accepted++
	fu.recordLocked(Event{Wall: wall, Timestamp: mm.Timestamp, SensorID: mm.SensorID,
		Kind: "accepted", NIS: out.NIS})
}

func (fu *Fuser) gateTrippedLocked(cs *channelState, nis float64) bool {
	gate := cs.cfg.DivergeNIS
	if gate <= 0 {
		gate = fu.cfg.Filter.NISGate
	}
	if gate <= 0 || math.IsNaN(nis) || math.IsInf(nis, 0) {
		return false
	}
	return nis > gate
}

// applyDueFaultsLocked moves non-finite fault timestamps up to t into each
// channel's bad streak and triggers isolation at the fault's own time.
func (fu *Fuser) applyDueFaultsLocked(t float64) {
	wall := time.Now()
	for _, cs := range fu.chans {
		var due, rest []float64
		for _, ft := range cs.pendingFaults {
			if ft <= t {
				due = append(due, ft)
			} else {
				rest = append(rest, ft)
			}
		}
		cs.pendingFaults = rest
		for _, ft := range due {
			cs.badStreak++
			cs.goodStreak = 0
			fu.maybeIsolateLocked(cs, ft, wall)
		}
	}
}

func (fu *Fuser) maybeIsolateLocked(cs *channelState, t float64, wall time.Time) {
	if cs.isolated || cs.badStreak < cs.cfg.DivergeCount {
		return
	}
	cs.isolated = true
	cs.isolatedAt = t
	cs.goodStreak = 0
	fu.stats.Isolations++
	fu.recordLocked(Event{Wall: wall, Timestamp: t, SensorID: cs.cfg.ID, Kind: "isolated"})
}

// scratchFilterLocked builds a throw-away filter holding a copy of the
// current state, used to trial-update isolated channels safely.
func (fu *Fuser) scratchFilterLocked() *filter.Filter {
	return fu.efilter.Clone()
}

// Estimate returns a copy of the current state.
func (fu *Fuser) Estimate() filter.State {
	fu.mu.Lock()
	defer fu.mu.Unlock()
	st := fu.efilter.State()
	if st == nil {
		return filter.State{}
	}
	return *st
}

// Stats returns a copy of the counters.
func (fu *Fuser) Stats() Stats {
	fu.mu.Lock()
	defer fu.mu.Unlock()
	return fu.stats
}

// Status returns a copy of all channel statuses.
func (fu *Fuser) Status() map[string]ChannelStatus {
	fu.mu.Lock()
	defer fu.mu.Unlock()
	out := make(map[string]ChannelStatus, len(fu.chans))
	for id, cs := range fu.chans {
		out[id] = ChannelStatus{
			ID: id, Isolated: cs.isolated,
			Accepted: cs.accepted, Rejected: cs.rejected,
			BadStreak: cs.badStreak, GoodStreak: cs.goodStreak,
			IsolatedAt: cs.isolatedAt, RecoveredAt: cs.recoveredAt,
		}
	}
	return out
}

// Events returns a copy of the decision log.
func (fu *Fuser) Events() []Event {
	fu.mu.Lock()
	defer fu.mu.Unlock()
	return fu.snapshotEventsLocked()
}

// Clock returns the current estimator time.
func (fu *Fuser) Clock() float64 {
	fu.mu.Lock()
	defer fu.mu.Unlock()
	return fu.clock
}

// Reset clears buffer, counters, events and isolation state and re-inits
// the filter to the seeded initial state.
func (fu *Fuser) Reset() {
	fu.mu.Lock()
	defer fu.mu.Unlock()
	for _, cs := range fu.chans {
		cs.isolated = false
		cs.accepted, cs.rejected = 0, 0
		cs.badStreak, cs.goodStreak = 0, 0
		cs.isolatedAt, cs.recoveredAt = 0, 0
		cs.pendingFaults = nil
	}
	fu.stats = Stats{}
	fu.log = nil
	fu.clock = fu.cfg.StartT
	fu.buf = alignment.NewBuffer(fu.cfg.Align)
	fu.efilter = filter.New(fu.efilter.Model(), fu.cfg.Filter)
	if fu.initX != nil {
		_ = fu.efilter.Init(fu.initedAt, fu.initX, fu.initP)
	}
}

func (fu *Fuser) recordLocked(e Event) {
	if e.Wall.IsZero() {
		e.Wall = time.Now()
	}
	fu.log = append(fu.log, e)
}

func (fu *Fuser) snapshotEventsLocked() []Event {
	out := make([]Event, len(fu.log))
	copy(out, fu.log)
	return out
}

func cloneSym(s *mat.SymDense) *mat.SymDense {
	n, _ := s.Dims()
	out := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			out.SetSym(i, j, s.At(i, j))
		}
	}
	return out
}
