// Session: long-lived, savable/resumable estimation sessions with historical
// backfill.
package engine

import (
	"errors"
	"io"
	"math"
	"sync"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/model"
)

// SessionConfig configures a savable Session.
type SessionConfig struct {
	// Engine is the underlying engine configuration (gate, horizon, ...).
	// It is recorded in every archive and must match exactly on restore,
	// because resuming a filter with a different gate/horizon/tolerance would
	// not reproduce bit-identical continuation.
	Engine Config
	// MemoryLimit bounds the retained, reconstructible operation history
	// (number of Submit/Advance records since the internal checkpoint). The
	// "have I processed this sample already" memory is bounded by the same
	// window: when the limit is exceeded the checkpoint folds forward and the
	// oldest processed-sample keys are forgotten. Measurements older than the
	// current floor are refused as expired_measurement, never fused twice.
	// <= 0 selects a built-in default.
	MemoryLimit int
}

// DefaultSessionConfig returns the default session configuration.
func DefaultSessionConfig() SessionConfig {
	return SessionConfig{Engine: DefaultConfig(), MemoryLimit: 4096}
}

// SourceBinding binds a source name to its measurement model and a
// self-describing, application-provided model descriptor. The descriptor is
// embedded verbatim in every archive and verified on restore, so a session
// cannot be resumed with an incompatible model for the same source name.
type SourceBinding struct {
	Source     string
	Descriptor string
	Model      model.MeasurementModel
}

// HistoricalMeasurement is one backfilled (historical) sample. Its model is
// resolved through the session's source bindings.
type HistoricalMeasurement struct {
	Time   float64
	Source string
	Z      *mat.Vector
}

type dedupKey struct {
	source string
	time   float64
}

// acceptedItem is one batch member that passed session-level partitioning and
// is being merged into the past.
type acceptedItem struct {
	key dedupKey
	h   HistoricalMeasurement
}

type opKind int

const (
	opSubmit opKind = iota
	opAdvance
)

// logOp is one reconstructible API call recorded since the checkpoint.
// Submit records are appended only when the sample passed every order-
// independent check and actually entered the engine's aligner; Advance
// records are strictly monotonic. Replaying the log on a scratch engine
// rebuilt from the checkpoint reproduces the current state bit-for-bit.
type logOp struct {
	kind   opKind
	source string
	time   float64
	z      *mat.Vector
	target float64
}

// Session is a concurrency-safe, persistable estimation session. It wraps an
// Engine and adds: (1) bounded memory of already processed measurements so
// duplicates are never fused twice; (2) historical backfill whose result is
// bit-identical to real-time arrival; and (3) self-describing, checksummed
// snapshots that can be resumed in a brand-new process.
type Session struct {
	mu         sync.Mutex
	cfg        SessionConfig
	motion     model.MotionModel
	motionDesc string
	models     map[string]model.MeasurementModel
	descs      map[string]string
	eng        *Engine
	clock      float64
	// cp is the folded-prefix baseline; replaying ops on an engine restored
	// from cp yields eng.
	cp   *engineSnapshot
	ops  []logOp
	seen map[dedupKey]struct{}
}

// NewSession constructs a session with the given initial state, motion model,
// motion descriptor and source bindings.
func NewSession(initial *model.State, motion model.MotionModel, motionDescriptor string, bindings []SourceBinding, cfg SessionConfig) (*Session, error) {
	if initial == nil {
		return nil, errors.New("session: nil initial state")
	}
	if motion == nil {
		return nil, errors.New("session: nil motion model")
	}
	if cfg.MemoryLimit <= 0 {
		cfg.MemoryLimit = DefaultSessionConfig().MemoryLimit
	}
	models := make(map[string]model.MeasurementModel, len(bindings))
	descs := make(map[string]string, len(bindings))
	for _, b := range bindings {
		if b.Source == "" {
			return nil, errors.New("session: binding with empty source name")
		}
		if b.Model == nil {
			return nil, errors.New("session: binding " + b.Source + " has nil model")
		}
		if _, dup := models[b.Source]; dup {
			return nil, errors.New("session: duplicate binding for source " + b.Source)
		}
		models[b.Source] = b.Model
		descs[b.Source] = b.Descriptor
	}
	eng, err := New(initial, motion, cfg.Engine)
	if err != nil {
		return nil, err
	}
	s := &Session{
		cfg:        cfg,
		motion:     motion,
		motionDesc: motionDescriptor,
		models:     models,
		descs:      descs,
		eng:        eng,
		clock:      initial.Time,
		seen:       make(map[dedupKey]struct{}),
	}
	s.cp = eng.snapshotInternal()
	return s, nil
}

func (s *Session) recordDirect(source string, t float64, reason RejectReason, detail string) *RejectError {
	re := reject(source, t, reason, detail)
	s.eng.externalReject(re)
	return re
}

// Submit offers one real-time measurement to the session. Order-independent
// structural problems and the order-independent parts of the session memory
// (already-processed duplicate, timestamp beyond the acceptable floor) are
// refused here; everything else is delegated to the deterministic engine.
func (s *Session) Submit(source string, t float64, z *mat.Vector) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	md := s.models[source]
	if source == "" || md == nil {
		return s.recordDirect(source, t, ReasonDimensionMismatch,
			"no model binding registered for source")
	}
	if math.IsNaN(t) || math.IsInf(t, 0) {
		return s.recordDirect(source, t, ReasonBadTimestamp, "timestamp is NaN/Inf")
	}
	if z == nil || z.Rows != md.Dim() || z.Cols != 1 {
		return s.recordDirect(source, t, ReasonDimensionMismatch,
			"measurement vector dimension != bound model dim")
	}
	// Expired takes precedence over duplicate (mirrors the aligner).
	if t < s.floor() {
		return s.recordDirect(source, t, ReasonExpired,
			"timestamp beyond the acceptable range (delay horizon or folded history)")
	}
	key := dedupKey{source, t}
	if _, dup := s.seen[key]; dup {
		return s.recordDirect(source, t, ReasonDuplicate,
			"sample was already processed by this session; refusing to fuse twice")
	}
	if err := s.eng.Submit(source, t, z.Clone(), md); err != nil {
		// Engine-side immediate rejection (structural): not logged, because it
		// never entered the aligner and replay reproduces it deterministically.
		return err
	}
	s.ops = append(s.ops, logOp{kind: opSubmit, source: source, time: t, z: z.Clone()})
	return nil
}

// floor is the oldest timestamp still usable: the later of "now minus delay
// horizon" and the folded-checkpoint time.
func (s *Session) floor() float64 {
	horizonFloor := s.clock - s.cfg.Engine.MaxDelayHorizon
	cpFloor := s.cp.State.Time
	if cpFloor > horizonFloor {
		return cpFloor
	}
	return horizonFloor
}

// Advance propagates the session to target time.
func (s *Session) Advance(target float64) (*model.State, []*RejectError, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if math.IsNaN(target) || math.IsInf(target, 0) {
		return nil, nil, errors.New("session: Advance target time is NaN/Inf")
	}
	if target < s.clock {
		return nil, nil, errors.New("session: Advance cannot move backwards in time")
	}
	// Every Advance is logged, including target == clock: a sample submitted
	// exactly at the committed clock can still become due, and replay must
	// reproduce that fusion.
	if target > s.clock {
		s.clock = target
	}
	_, rejs, due, err := s.eng.advanceInternal(target)
	if err != nil {
		return nil, nil, err
	}
	s.ops = append(s.ops, logOp{kind: opAdvance, target: target})
	s.rememberDue(due)
	s.foldIfNeeded()
	return s.eng.stateInternal(), rejs, nil
}

func (s *Session) rememberDue(due []*Measurement) {
	for _, m := range due {
		s.seen[dedupKey{m.Source, m.Time}] = struct{}{}
	}
}

// foldIfNeeded keeps the retained operation log bounded. Each fold replays a
// prefix up to an Advance boundary onto a scratch engine restored from the
// checkpoint, promotes that scratch to the checkpoint and forgets the
// processed-sample keys of the folded prefix. Folding is repeated until the
// retained log fits the limit (or no further Advance boundary exists), so the
// retained memory has a hard upper bound even after a very large backfill
// batch. Amortized cost is linear in the limit and happens while s.mu is held.
func (s *Session) foldIfNeeded() {
	for len(s.ops) > s.cfg.MemoryLimit {
		// Keep the most recent limit ops; find the first Advance boundary at
		// or before them so the prefix ends on a self-contained step.
		cut := len(s.ops) - s.cfg.MemoryLimit
		if cut < 0 {
			cut = 0
		}
		for cut < len(s.ops) && s.ops[cut].kind != opAdvance {
			cut++
		}
		cut++ // keep the boundary advance inside the folded prefix
		if cut >= len(s.ops) {
			// The tail contains no further Advance boundary: every retained op
			// is still reconstructible from the checkpoint at the next fold.
			return
		}
		s.foldPrefix(cut)
	}
}

// foldPrefix replays ops[:cut] from the checkpoint, promotes the resulting
// scratch engine to checkpoint and removes those ops plus their due keys.
func (s *Session) foldPrefix(cut int) {
	scratch, err := restoreEngine(s.cp, s.motion, s.models)
	if err != nil {
		panic("session: cannot restore own checkpoint: " + err.Error())
	}
	drop := make(map[dedupKey]struct{})
	for _, op := range s.ops[:cut] {
		switch op.kind {
		case opSubmit:
			_ = scratch.Submit(op.source, op.time, op.z, s.models[op.source])
		case opAdvance:
			_, _, due, err := scratch.advanceInternal(op.target)
			if err != nil {
				panic("session: checkpoint replay failed: " + err.Error())
			}
			for _, m := range due {
				drop[dedupKey{m.Source, m.Time}] = struct{}{}
			}
		}
	}
	s.cp = scratch.snapshotInternal()
	s.ops = append([]logOp(nil), s.ops[cut:]...)
	for k := range drop {
		delete(s.seen, k)
	}
}

// Backfill merges a batch of historical measurements into the past. The final
// estimate is bit-identical to having received the accepted samples in real
// time. Samples are partitioned, without being applied, into:
//   - fused: structurally valid, within the acceptable floor, not duplicate;
//   - rejected expired: timestamp older than the acceptable range;
//   - rejected duplicate: already processed (in a previous live run, a prior
//     backfill batch, or appearing more than once within this batch).
//
// The batch is atomic with respect to validation: if any item is structurally
// malformed (unknown source, NaN/Inf/future timestamp, bad dimension), no item
// is applied and an error is returned.
func (s *Session) Backfill(batch []HistoricalMeasurement) (fused int, rejections []*RejectError, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Phase 1: whole-batch structural validation (nothing is applied).
	for _, h := range batch {
		md := s.models[h.Source]
		if md == nil {
			return 0, nil, errors.New("session.Backfill: item has no registered model binding")
		}
		if math.IsNaN(h.Time) || math.IsInf(h.Time, 0) {
			return 0, nil, errors.New("session.Backfill: item timestamp is NaN/Inf")
		}
		if h.Time > s.clock {
			return 0, nil, errors.New("session.Backfill: historical item lies in the future of the session clock")
		}
		if h.Z == nil || h.Z.Rows != md.Dim() || h.Z.Cols != 1 {
			return 0, nil, errors.New("session.Backfill: item vector dimension != bound model dim")
		}
	}

	// Phase 2: order-independent partition. Duplicates inside THIS batch form
	// a duplicate group and every member is refused, mirroring aligner
	// semantics (no copy can be declared authoritative).
	batchCount := make(map[dedupKey]int, len(batch))
	for _, h := range batch {
		batchCount[dedupKey{h.Source, h.Time}]++
	}
	var accepted []acceptedItem
	rejections = make([]*RejectError, 0)
	for _, h := range batch {
		key := dedupKey{h.Source, h.Time}
		if h.Time < s.floor() {
			re := reject(h.Source, h.Time, ReasonExpired,
				"historical sample beyond the acceptable range (delay horizon or folded history)")
			rejections = append(rejections, re)
			continue
		}
		if batchCount[key] > 1 {
			re := reject(h.Source, h.Time, ReasonDuplicate,
				"historical sample appears more than once in the same backfill batch; all copies refused")
			rejections = append(rejections, re)
			continue
		}
		if _, dup := s.seen[key]; dup {
			re := reject(h.Source, h.Time, ReasonDuplicate,
				"historical sample was already processed (live or an earlier backfill); refusing to fuse twice")
			rejections = append(rejections, re)
			continue
		}
		accepted = append(accepted, acceptedItem{key: key, h: h})
	}

	// Phase 3: rebuild the engine as if the accepted samples had been
	// available since the checkpoint, then replay the retained operation log.
	scratch, rerr := restoreEngine(s.cp, s.motion, s.models)
	if rerr != nil {
		panic("session: cannot restore own checkpoint: " + rerr.Error())
	}
	// Prefuse order is (time, source) for stable buffering; the aligner
	// re-sorts at every advance, so processing order is independent of it.
	sortAccepted(accepted)
	acceptedKeys := make(map[dedupKey]struct{}, len(accepted))
	for _, a := range accepted {
		acceptedKeys[a.key] = struct{}{}
		_ = scratch.Submit(a.h.Source, a.h.Time, a.h.Z.Clone(), s.models[a.h.Source])
	}
	newSeen := make(map[dedupKey]struct{}, len(s.seen)+len(accepted))
	for _, op := range s.ops {
		switch op.kind {
		case opSubmit:
			_ = scratch.Submit(op.source, op.time, op.z, s.models[op.source])
		case opAdvance:
			_, _, due, ferr := scratch.advanceInternal(op.target)
			if ferr != nil {
				return 0, nil, ferr
			}
			for _, m := range due {
				newSeen[dedupKey{m.Source, m.Time}] = struct{}{}
			}
		}
	}

	// Pre-phase rejects (expired/duplicate decided at the session level) are
	// applied to the rebuilt engine after replay so its statistics count them,
	// exactly as the real-time reference engine would count them.
	var directRejects []*RejectError
	for _, re := range rejections {
		directRejects = append(directRejects, re)
	}

	// Collect rejects that the rebuilt history attributes to accepted batch
	// items (out_of_gate, stream_faulty, non_finite_value, ...).
	allRej := scratch.rejectionsInternal()
	for _, re := range allRej {
		if _, isBatch := acceptedKeys[dedupKey{re.Source, re.Time}]; isBatch {
			rejections = append(rejections, re)
		}
	}
	for _, re := range directRejects {
		scratch.externalReject(re)
	}

	// Commit the rebuilt world. Accepted samples become retained submit ops
	// placed BEFORE the logged ops, exactly matching the replay ordering.
	newOps := make([]logOp, 0, len(accepted)+len(s.ops))
	for _, a := range accepted {
		newOps = append(newOps, logOp{kind: opSubmit, source: a.h.Source, time: a.h.Time, z: a.h.Z.Clone()})
	}
	newOps = append(newOps, s.ops...)

	s.eng = scratch
	s.ops = newOps
	s.seen = newSeen
	s.foldIfNeeded()

	batchRej := 0
	for _, re := range rejections {
		if _, isBatch := acceptedKeys[dedupKey{re.Source, re.Time}]; isBatch {
			batchRej++
		}
	}
	fused = len(accepted) - batchRej
	return fused, rejections, nil
}

func sortAccepted(items []acceptedItem) {
	// Implemented as a simple insertion sort: batches are short and this keeps
	// imports minimal.
	for i := 1; i < len(items); i++ {
		for j := i; j > 0; j-- {
			a, b := items[j-1], items[j]
			if b.key.time < a.key.time || (b.key.time == a.key.time && b.key.source < a.key.source) {
				items[j-1], items[j] = b, a
			} else {
				break
			}
		}
	}
}

// State returns a snapshot copy of the current state.
func (s *Session) State() *model.State {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.eng.State()
}

// Stats returns a copy of the statistics snapshot.
func (s *Session) Stats() Stats {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.eng.Stats()
}

// Rejections returns every recorded rejection so far (copied).
func (s *Session) Rejections() []*RejectError {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.eng.Rejections()
}

// Snapshot exports a self-describing, checksummed archive of the session.
func (s *Session) Snapshot() ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.encodeLocked()
}

// Save writes the archive to w.
func (s *Session) Save(w io.Writer) error {
	data, err := s.Snapshot()
	if err != nil {
		return err
	}
	_, err = w.Write(data)
	return err
}

// RestoreSession reads an archive produced by Snapshot/Save and resumes it.
// The motion model/descriptor and source bindings must match the descriptors
// recorded in the archive, and the engine configuration must match the one
// recorded; otherwise ErrArchiveBinding is returned and no partial session is
// constructed.
func RestoreSession(r io.Reader, motion model.MotionModel, motionDescriptor string, bindings []SourceBinding, cfg SessionConfig) (*Session, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return restoreSessionBytes(data, motion, motionDescriptor, bindings, cfg)
}
