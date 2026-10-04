package engine

import (
	"errors"
	"math"
	"sync"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/model"
)

// Config configures the concurrent estimation engine.
type Config struct {
	// MaxDelayHorizon is how far in the past a measurement may be relative
	// to the latest committed time and still be accepted (seconds, >= 0).
	MaxDelayHorizon float64
	// EigTol is the relative tolerance for symmetry/PSD checks (e.g. 1e-9).
	EigTol float64
	// CondLimit rejects updates whose innovation covariance condition
	// number exceeds it (e.g. 1e12).
	CondLimit float64
	// Health tunes NIS gating and fault/recovery behavior.
	Health HealthConfig
}

// DefaultConfig returns conservative numerically safe defaults.
func DefaultConfig() Config {
	return Config{
		MaxDelayHorizon: 2.0,
		EigTol:          1e-9,
		CondLimit:       1e12,
		Health:          DefaultHealthConfig(),
	}
}

// StreamStats holds per-stream counters.
type StreamStats struct {
	Accepted      int
	Rejected      map[RejectReason]int
	HardFaults    int
	SoftFaults    int
	Recoveries    int
	CurrentlyDown bool
}

// Stats is a snapshot of engine statistics.
type Stats struct {
	Predicts  int
	Updates   int
	Rejected  int
	PerStream map[string]StreamStats
}

// Engine is the concurrency-safe facade over EKF + aligner + health tracker.
// A single mutex guards every state transition; the deterministic processing
// order is (time, source), so concurrent submissions always converge to the
// same final state.
type Engine struct {
	mu       sync.Mutex
	cfg      Config
	f        *EKF
	al       *aligner
	ht       *healthTracker
	stats    Stats
	rejected []*RejectError
}

// New constructs an engine. The initial covariance must be symmetric PSD.
func New(initial *model.State, motion model.MotionModel, cfg Config) (*Engine, error) {
	d := DefaultConfig()
	if cfg.MaxDelayHorizon < 0 || math.IsNaN(cfg.MaxDelayHorizon) || math.IsInf(cfg.MaxDelayHorizon, 0) {
		return nil, errors.New("engine: MaxDelayHorizon must be finite and >= 0")
	}
	if cfg.EigTol <= 0 || math.IsNaN(cfg.EigTol) {
		cfg.EigTol = d.EigTol
	}
	if cfg.CondLimit <= 0 || math.IsNaN(cfg.CondLimit) || math.IsInf(cfg.CondLimit, 0) {
		cfg.CondLimit = d.CondLimit
	}
	if cfg.Health.GateSigma <= 0 || cfg.Health.FaultStreak <= 0 || cfg.Health.RecoverStreak <= 0 {
		cfg.Health = d.Health
	}
	f, err := NewEKF(initial, motion, cfg.EigTol, cfg.CondLimit)
	if err != nil {
		return nil, err
	}
	return &Engine{
		cfg:   cfg,
		f:     f,
		al:    newAligner(initial.Time, cfg.MaxDelayHorizon),
		ht:    newHealthTracker(cfg.Health),
		stats: Stats{PerStream: make(map[string]StreamStats)},
	}, nil
}

func (e *Engine) streamStat(source string) StreamStats {
	s, found := e.stats.PerStream[source]
	if !found {
		s = StreamStats{}
	}
	if s.Rejected == nil {
		s.Rejected = make(map[RejectReason]int)
	}
	s.CurrentlyDown = e.ht.faulty(source)
	return s
}

func (e *Engine) recordReject(re *RejectError) {
	s := e.streamStat(re.Source)
	s.Rejected[re.Reason]++
	s.CurrentlyDown = e.ht.faulty(re.Source)
	e.stats.PerStream[re.Source] = s
	e.stats.Rejected++
	e.rejected = append(e.rejected, re)
}

// Submit enqueues a measurement from a source. Only order-independent
// structural checks happen here (empty source, nil model, wrong dimensions,
// NaN/Inf timestamp, exact duplicate, expired). Non-finite measurement VALUES
// are still buffered: stream isolation is decided in deterministic
// (time, source) order during Advance, which is what makes results
// reproducible under concurrent arrival. No measurement is dropped silently.
func (e *Engine) Submit(source string, t float64, z *mat.Vector, md model.MeasurementModel) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if source == "" {
		re := reject(source, t, ReasonDimensionMismatch, "empty source name")
		e.recordReject(re)
		return re
	}
	if md == nil {
		re := reject(source, t, ReasonDimensionMismatch, "nil measurement model")
		e.recordReject(re)
		return re
	}
	if z == nil || z.Rows != md.Dim() || z.Cols != 1 {
		re := reject(source, t, ReasonDimensionMismatch, "measurement vector dimension != model dim")
		e.recordReject(re)
		return re
	}
	// Non-finite values are intentionally buffered, not rejected here: the
	// hard-fault decision happens in time order inside Advance, independent
	// of which goroutine delivered the sample.

	m := &Measurement{Time: t, Source: source, Z: z.Clone(), Model: md}
	if err := e.al.submit(m); err != nil {
		var re *RejectError
		if errors.As(err, &re) {
			e.recordReject(re)
			return re
		}
		return err
	}
	return nil
}

// mapUpdateError converts a low-level EKF error into a typed *RejectError.
func mapUpdateError(source string, t float64, err error) *RejectError {
	detail := err.Error()
	switch {
	case errors.Is(err, ErrNonFinite):
		return reject(source, t, ReasonNonFinite, detail)
	case errors.Is(err, ErrIllegalNoise):
		return reject(source, t, ReasonIllegalNoise, detail)
	case errors.Is(err, ErrDimensionMismatch):
		return reject(source, t, ReasonDimensionMismatch, detail)
	case errors.Is(err, ErrDegenerate):
		return reject(source, t, ReasonDegenerate, detail)
	case errors.Is(err, ErrIllConditioned):
		return reject(source, t, ReasonIllConditioned, detail)
	case errors.Is(err, ErrBadModel):
		return reject(source, t, ReasonModelFailed, detail)
	default:
		return reject(source, t, ReasonModelFailed, detail)
	}
}

// probeAssess evaluates an isolated stream's measurement WITHOUT mutating the
// EKF: it runs the measurement model and computes NIS against the current
// predicted state. Good (finite, evaluable, in-gate) trials advance recovery;
// anything bad resets it. The measurement itself is recorded as rejected with
// ReasonStreamFaulty while isolation lasts.
func (e *Engine) probeAssess(m *Measurement, newRejections *[]*RejectError) {
	source := m.Source
	good := false
	// Evaluate the isolated stream's sample read-only against the current
	// predicted state; it never mutates the global estimate.
	nis, ok := e.computeNIS(m)
	good = ok && nis <= e.cfg.Health.GateSigma*e.cfg.Health.GateSigma

	re := reject(source, m.Time, ReasonStreamFaulty, "stream isolated; sample used as recovery probe but not fused")
	e.recordReject(re)
	*newRejections = append(*newRejections, re)

	recovered := e.ht.Probe(source, good)
	if recovered {
		s := e.streamStat(source)
		s.Recoveries++
		s.CurrentlyDown = false
		e.stats.PerStream[source] = s
	}
}

// computeNIS evaluates y^T S^{-1} y read-only against the current filter
// state. ok=false covers any structural/numerical failure.
func (e *Engine) computeNIS(m *Measurement) (float64, bool) {
	return e.f.NIS(m.Z, m.Model)
}

// processOne fuses one due measurement from a healthy stream, or treats an
// isolated stream's sample as a non-polluting recovery probe.
func (e *Engine) processOne(m *Measurement, newRejections *[]*RejectError) {
	source := m.Source

	if !e.ht.healthy(source) {
		e.probeAssess(m, newRejections)
		return
	}

	// Non-finite value arriving in deterministic time order: hard-fault the
	// stream now, isolate it, do not touch the estimate.
	if !mat.Finite(m.Z) {
		e.ht.markHardFault(source)
		s := e.streamStat(source)
		s.HardFaults++
		s.CurrentlyDown = true
		e.stats.PerStream[source] = s
		re := reject(source, m.Time, ReasonNonFinite,
			"measurement contains NaN/+Inf/-Inf; stream isolated, state untouched")
		e.recordReject(re)
		*newRejections = append(*newRejections, re)
		return
	}

	// Gate first with a read-only NIS so an implausible outlier never
	// mutates the global estimate.
	nis, ok := e.computeNIS(m)
	gate := e.cfg.Health.GateSigma * e.cfg.Health.GateSigma
	if !ok {
		// Read-only assessment failed (illegal noise, degeneracy, ...); apply
		// the same validation through Update to produce a typed rejection.
		if _, err := e.f.Update(m.Z, m.Model); err != nil {
			re := mapUpdateError(source, m.Time, err)
			hard := re.Reason == ReasonNonFinite
			s := e.streamStat(source)
			if hard {
				s.HardFaults++
				e.ht.markHardFault(source)
			}
			s.CurrentlyDown = e.ht.faulty(source)
			e.stats.PerStream[source] = s
			e.recordReject(re)
			*newRejections = append(*newRejections, re)
			return
		}
		return
	}

	if nis > gate {
		// In-gate failure: do NOT fuse. Count streak; isolate if persistent.
		s := e.streamStat(source)
		s.SoftFaults++
		isolated := e.ht.Observe(source, false, true)
		s.CurrentlyDown = isolated
		e.stats.PerStream[source] = s
		var re *RejectError
		if isolated {
			re = reject(source, m.Time, ReasonStreamFaulty,
				"NIS exceeded innovation gate for too many consecutive samples; stream isolated")
		} else {
			re = reject(source, m.Time, ReasonOutOfGate,
				"NIS exceeded innovation gate; sample withheld from estimate")
		}
		e.recordReject(re)
		*newRejections = append(*newRejections, re)
		return
	}

	// In-gate: perform the actual mutating update.
	nis, err := e.f.Update(m.Z, m.Model)
	if err != nil {
		// Should be rare after a successful read-only NIS; treat uniformly.
		re := mapUpdateError(source, m.Time, err)
		e.recordReject(re)
		*newRejections = append(*newRejections, re)
		return
	}
	_ = nis

	if isolated := e.ht.Observe(source, false, false); isolated {
		// Defensive: cannot transition healthy->faulty on a good sample.
	}
	s := e.streamStat(source)
	s.Accepted++
	s.CurrentlyDown = false
	e.stats.Updates++
	e.stats.PerStream[source] = s
}

// Advance propagates the filter to target time, fusing every measurement due
// by then in deterministic (time, source) order, and returns the rejections
// produced while processing those measurements.
func (e *Engine) Advance(target float64) (*model.State, []*RejectError, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	st, rejs, _, err := e.advanceLocked(target)
	return st, rejs, err
}

// advanceLocked is the lock-free body of Advance. It additionally returns the
// due measurements that were actually processed (fused or evaluated as
// probes) at this step, which lets the Session layer maintain its bounded
// memory of processed samples. Caller must hold e.mu.
func (e *Engine) advanceLocked(target float64) (*model.State, []*RejectError, []*Measurement, error) {
	if math.IsNaN(target) || math.IsInf(target, 0) {
		return nil, nil, nil, errors.New("engine: Advance target time is NaN/Inf")
	}
	if target < e.f.Time() {
		return nil, nil, nil, errors.New("engine: Advance cannot move backwards in time")
	}

	newRejections := make([]*RejectError, 0)
	due, alignRejects := e.al.advance(target)
	for _, re := range alignRejects {
		e.recordReject(re)
		newRejections = append(newRejections, re)
	}

	// Predict lazily to each measurement time, then finally to target, so
	// delayed-but-valid measurements are fused at their own timestamp.
	for _, m := range due {
		if m.Time > e.f.Time() {
			if err := e.f.Predict(m.Time); err != nil {
				return nil, nil, nil, err
			}
			e.stats.Predicts++
		}
		e.processOne(m, &newRejections)
	}

	if target > e.f.Time() {
		if err := e.f.Predict(target); err != nil {
			return nil, nil, nil, err
		}
		e.stats.Predicts++
	}

	return e.f.State(), newRejections, due, nil
}

// State returns a snapshot copy of the current state.
func (e *Engine) State() *model.State {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.f.State()
}

// Stats returns a copy of the statistics snapshot.
func (e *Engine) Stats() Stats {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := Stats{
		Predicts:  e.stats.Predicts,
		Updates:   e.stats.Updates,
		Rejected:  e.stats.Rejected,
		PerStream: make(map[string]StreamStats, len(e.stats.PerStream)),
	}
	for k, v := range e.stats.PerStream {
		cp := StreamStats{
			Accepted:      v.Accepted,
			Rejected:      make(map[RejectReason]int, len(v.Rejected)),
			HardFaults:    v.HardFaults,
			SoftFaults:    v.SoftFaults,
			Recoveries:    v.Recoveries,
			CurrentlyDown: v.CurrentlyDown,
		}
		for rk, rv := range v.Rejected {
			cp.Rejected[rk] = rv
		}
		out.PerStream[k] = cp
	}
	return out
}

// Rejections returns every recorded rejection so far (copied).
func (e *Engine) Rejections() []*RejectError {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([]*RejectError, len(e.rejected))
	copy(out, e.rejected)
	return out
}
