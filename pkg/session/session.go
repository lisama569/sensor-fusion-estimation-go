package session

import (
	"sync"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/engine"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/model"
)

// Options configures a long-lived session.
type Options struct {
	// MaxProcessedPerSource bounds the processed-measurement memory per source
	// (hard cap; <= 0 selects DefaultMaxProcessedPerSource).
	MaxProcessedPerSource int
	// Engine optionally overrides engine configuration. Its MaxDelayHorizon
	// must agree with the session horizon used for memory pruning.
	Engine engine.Config
}

// DefaultOptions returns conservative defaults (engine defaults).
func DefaultOptions() Options {
	return Options{Engine: engine.DefaultConfig(), MaxProcessedPerSource: DefaultMaxProcessedPerSource}
}

// inflightKey identifies one submitted-but-not-yet-resolved measurement.
type inflightKey struct {
	source string
	time   float64
}

// Session is a long-lived, resumable estimation session. It wraps an engine
// and adds cross-archive/cross-batch duplicate rejection, historical replay
// and versioned persistence.
//
// A single session mutex guards every transition; the wrapped engine keeps its
// own lock. Because all order-sensitive decisions still happen inside the
// engine's deterministic (time, source) order, concurrent submissions of the
// same multiset produce bit-identical results, and every export represents one
// consistent point in time.
type Session struct {
	mu      sync.Mutex
	eng     *engine.Engine
	mem     *memory
	horizon float64
	// inflight holds keys submitted but not yet released by an Advance. They
	// are deliberately NOT in permanent memory yet, so two copies of one
	// measurement inside the same batch reach the engine and receive the
	// engine's duplicate_timestamp rejection (distinguishable from a
	// cross-batch/cross-archive already_processed refusal).
	inflight map[inflightKey]int
}

// New creates a session from an initial state and motion model.
func New(initial *model.State, motion model.MotionModel, opts Options) (*Session, error) {
	cfg := opts.Engine
	// Mirror the engine's own default filling so the session horizon matches.
	def := engine.DefaultConfig()
	if cfg.MaxDelayHorizon < 0 || cfg.MaxDelayHorizon == 0 {
		cfg.MaxDelayHorizon = def.MaxDelayHorizon
	}
	if cfg.EigTol <= 0 {
		cfg.EigTol = def.EigTol
	}
	if cfg.CondLimit <= 0 {
		cfg.CondLimit = def.CondLimit
	}
	if cfg.Health.GateSigma <= 0 || cfg.Health.FaultStreak <= 0 || cfg.Health.RecoverStreak <= 0 {
		cfg.Health = def.Health
	}
	eng, err := engine.New(initial, motion, cfg)
	if err != nil {
		return nil, err
	}
	return &Session{
		eng:      eng,
		mem:      newMemory(cfg.MaxDelayHorizon, opts.MaxProcessedPerSource),
		horizon:  cfg.MaxDelayHorizon,
		inflight: make(map[inflightKey]int),
	}, nil
}

// Submit enqueues one measurement. It refuses, with the distinguishable reason
// already_processed, an exact (source, timestamp) pair that was already
// resolved (fused or otherwise decided) before an archive boundary or in an
// earlier replay batch; such a sample can never reach the engine and is
// therefore never fused twice. Structural validation and all other reject
// reasons (including same-batch duplicate_timestamp and expired_measurement)
// are delegated to the engine exactly as in an uninterrupted run.
func (s *Session) Submit(source string, t float64, z *mat.Vector, md model.MeasurementModel) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.submitLocked(source, t, z, md)
}

// submitLocked is Submit's body without locking (callers hold s.mu).
func (s *Session) submitLocked(source string, t float64, z *mat.Vector, md model.MeasurementModel) error {
	// Permanent memory only covers measurements already resolved in an earlier
	// Advance / before an archive boundary. In-flight (buffered) copies are
	// intentionally absent so same-batch duplicates are classified by engine.
	if s.mem.seen(source, t) {
		// Expiry takes precedence when the timestamp is unusable at all: the
		// engine records the canonical expired_measurement rejection, keeping
		// the reasons distinguishable.
		if t < s.nowLocked()-s.horizon {
			err := s.eng.Submit(source, t, z, md)
			s.trackRejectedSubmission(source, t, err)
			return err
		}
		return engine.Reject(source, t, engine.ReasonAlreadyProcessed,
			"measurement already resolved before an archive boundary or in an earlier replay batch; refusing to fuse twice")
	}
	err := s.eng.Submit(source, t, z, md)
	if err == nil {
		s.inflight[inflightKey{source, t}]++
		return nil
	}
	// Structurally refused at submission (bad timestamp/dimension/...): the
	// engine already recorded it; nothing can ever come due for this key.
	s.trackRejectedSubmission(source, t, err)
	return err
}

// trackRejectedSubmission handles the bookkeeping for a Submit that failed
// synchronously. Most structural failures are order-independent rejects; the
// only key-related case we must avoid marking is nothing here (permanent
// membership is committed in advanceLocked).
func (s *Session) trackRejectedSubmission(source string, t float64, err error) {
	_ = source
	_ = t
	_ = err
}

func (s *Session) nowLocked() float64 { return s.eng.State().Time }

// ReplayItem is one measurement in an offline backfill batch.
type ReplayItem struct {
	Source string
	Time   float64
	Z      *mat.Vector
	Model  model.MeasurementModel
}

// Replay backfills one historical batch: items are submitted (refusing
// cross-batch/cross-archive duplicates as already_processed) and then fused in
// the engine's deterministic (time, source) order up to the newest accepted
// item time. The outcome is bit-identical to those measurements having arrived
// live, provided they are replayed before the clock passes their timestamps.
// It returns every rejection produced by the batch.
func (s *Session) Replay(items []ReplayItem) ([]*engine.RejectError, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := make([]*engine.RejectError, 0)
	maxT := s.nowLocked()
	for _, it := range items {
		err := s.submitLocked(it.Source, it.Time, it.Z, it.Model)
		if err != nil {
			if re := asReject(err); re != nil {
				out = append(out, re)
				continue
			}
			return out, err
		}
		if it.Time > maxT {
			maxT = it.Time
		}
	}
	_, rej, err := s.advanceLocked(maxT)
	if err != nil {
		return out, err
	}
	out = append(out, rej...)
	return out, nil
}

// advanceLocked runs one engine Advance, commits the now-resolved measurements
// to permanent processed-memory, and prunes that memory to the new time.
func (s *Session) advanceLocked(target float64) (*model.State, []*engine.RejectError, error) {
	st, rej, err := s.eng.Advance(target)
	if err != nil {
		return nil, nil, err
	}

	// Which in-flight keys were refused as same-batch duplicates? All members
	// of a duplicate group are refused and must NOT be marked permanently
	// processed: the timestamp never got an authoritative decision, matching
	// the uninterrupted engine semantics.
	dup := make(map[inflightKey]struct{})
	for _, r := range rej {
		if r != nil && r.Reason == engine.ReasonDuplicate {
			dup[inflightKey{r.Source, r.Time}] = struct{}{}
		}
	}

	// Every submitted key whose timestamp is now released (<= target) got its
	// deterministic decision (fused, gated, isolated-probe, or rejected for any
	// non-duplicate reason). Commit it to permanent memory so it can never be
	// fused again in a later batch/archive.
	for k := range s.inflight {
		if k.time > target {
			continue // still buffered for a future Advance
		}
		delete(s.inflight, k)
		if _, isDup := dup[k]; isDup {
			continue
		}
		s.mem.add(k.source, k.time)
	}

	s.mem.advance(st.Time)
	return st, rej, nil
}

// Advance advances the live clock and fuses due measurements in deterministic
// order. It never moves backwards.
func (s *Session) Advance(target float64) (*model.State, []*engine.RejectError, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.advanceLocked(target)
}

// State returns a snapshot copy of the current state.
func (s *Session) State() *model.State {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.eng.State()
}

// Stats returns the engine statistics snapshot.
func (s *Session) Stats() engine.Stats {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.eng.Stats()
}

// Rejections returns every recorded rejection so far (copied).
func (s *Session) Rejections() []*engine.RejectError {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.eng.Rejections()
}

// Engine exposes the wrapped engine for callers that need lower-level access
// (read-only snapshots such as State/Stats remain concurrency-safe).
func (s *Session) Engine() *engine.Engine {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.eng
}

func asReject(err error) *engine.RejectError {
	if err == nil {
		return nil
	}
	if re, ok := err.(*engine.RejectError); ok {
		return re
	}
	return nil
}
