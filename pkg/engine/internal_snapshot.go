package engine

import (
	"errors"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/model"
)

// engineSnapshot is a plain-data, fully deep-copied view of every mutable
// part of an Engine. Session archives are built from it, and replay
// (backfill) restores scratch engines through it. It lives inside package
// engine so it can touch unexported fields; persistence (encoding, checksums,
// versioning) is implemented in archive.go.
type engineSnapshot struct {
	CFG        Config
	State      *model.State
	AlignerNow float64
	Pending    []pendingMeasurement
	Committed  map[string]float64
	Health     map[string]streamHealthData
	Stats      Stats
	Rejected   []rejectionData
}

// pendingMeasurement is the serializable form of an aligner-pending sample.
// Models are re-resolved from the session's bindings on restore, so no
// interface is carried here.
type pendingMeasurement struct {
	Time   float64
	Source string
	Z      *mat.Vector
}

// streamHealthData is the plain-data state machine of one stream.
type streamHealthData struct {
	Faulty     bool
	BadStreak  int
	GoodStreak int
}

// rejectionData is the serializable form of *RejectError.
type rejectionData struct {
	Source string
	Time   float64
	Reason RejectReason
	Detail string
}

// snapshotLocked returns a fully independent copy of the engine internals.
// Caller must hold e.mu.
func (e *Engine) snapshotLocked() *engineSnapshot {
	sn := &engineSnapshot{
		CFG:        e.cfg,
		State:      e.f.State(),
		AlignerNow: e.al.now,
		Pending:    make([]pendingMeasurement, 0, len(e.al.pending)),
		Committed:  make(map[string]float64, len(e.al.committed)),
		Health:     make(map[string]streamHealthData, len(e.ht.streams)),
		Stats: Stats{
			Predicts:  e.stats.Predicts,
			Updates:   e.stats.Updates,
			Rejected:  e.stats.Rejected,
			PerStream: make(map[string]StreamStats, len(e.stats.PerStream)),
		},
		Rejected: make([]rejectionData, 0, len(e.rejected)),
	}
	for _, m := range e.al.pending {
		sn.Pending = append(sn.Pending, pendingMeasurement{
			Time: m.Time, Source: m.Source, Z: m.Z.Clone(),
		})
	}
	for k, v := range e.al.committed {
		sn.Committed[k] = v
	}
	for k, v := range e.ht.streams {
		sn.Health[k] = streamHealthData{Faulty: v.faulty, BadStreak: v.badStreak, GoodStreak: v.goodStreak}
	}
	for k, v := range e.stats.PerStream {
		cp := StreamStats{
			Accepted:      v.Accepted,
			HardFaults:    v.HardFaults,
			SoftFaults:    v.SoftFaults,
			Recoveries:    v.Recoveries,
			CurrentlyDown: v.CurrentlyDown,
			Rejected:      make(map[RejectReason]int, len(v.Rejected)),
		}
		for rk, rv := range v.Rejected {
			cp.Rejected[rk] = rv
		}
		sn.Stats.PerStream[k] = cp
	}
	for _, re := range e.rejected {
		sn.Rejected = append(sn.Rejected, rejectionData{
			Source: re.Source, Time: re.Time, Reason: re.Reason, Detail: re.Detail,
		})
	}
	return sn
}

// restoreEngine builds a fresh Engine from a snapshot, resolving pending
// measurement models through modelsBySource. The motion model is supplied
// separately because it is never serialized.
func restoreEngine(sn *engineSnapshot, motion model.MotionModel, modelsBySource map[string]model.MeasurementModel) (*Engine, error) {
	if sn == nil || sn.State == nil {
		return nil, errors.New("restore: empty engine snapshot")
	}
	if motion == nil {
		return nil, errors.New("restore: nil motion model")
	}
	f, err := NewEKF(sn.State, motion, sn.CFG.EigTol, sn.CFG.CondLimit)
	if err != nil {
		return nil, err
	}
	e := &Engine{
		cfg: sn.CFG,
		f:   f,
		al:  newAligner(sn.AlignerNow, sn.CFG.MaxDelayHorizon),
		ht:  newHealthTracker(sn.CFG.Health),
		stats: Stats{
			Predicts:  sn.Stats.Predicts,
			Updates:   sn.Stats.Updates,
			Rejected:  sn.Stats.Rejected,
			PerStream: make(map[string]StreamStats, len(sn.Stats.PerStream)),
		},
		rejected: make([]*RejectError, 0, len(sn.Rejected)),
	}
	e.al.committed = make(map[string]float64, len(sn.Committed))
	for k, v := range sn.Committed {
		e.al.committed[k] = v
	}
	e.al.pending = make([]*Measurement, 0, len(sn.Pending))
	for _, p := range sn.Pending {
		md := modelsBySource[p.Source]
		if md == nil {
			return nil, errors.New("restore: no model bound for pending source " + p.Source)
		}
		e.al.pending = append(e.al.pending, &Measurement{Time: p.Time, Source: p.Source, Z: p.Z.Clone(), Model: md})
	}
	for k, v := range sn.Health {
		e.ht.streams[k] = &streamHealth{faulty: v.Faulty, badStreak: v.BadStreak, goodStreak: v.GoodStreak}
	}
	for k, v := range sn.Stats.PerStream {
		cp := StreamStats{
			Accepted:      v.Accepted,
			HardFaults:    v.HardFaults,
			SoftFaults:    v.SoftFaults,
			Recoveries:    v.Recoveries,
			CurrentlyDown: v.CurrentlyDown,
			Rejected:      make(map[RejectReason]int, len(v.Rejected)),
		}
		for rk, rv := range v.Rejected {
			cp.Rejected[rk] = rv
		}
		e.stats.PerStream[k] = cp
	}
	for _, rd := range sn.Rejected {
		e.rejected = append(e.rejected, &RejectError{Source: rd.Source, Time: rd.Time, Reason: rd.Reason, Detail: rd.Detail})
	}
	return e, nil
}

// snapshotInternal returns a deep copy of engine internals.
func (e *Engine) snapshotInternal() *engineSnapshot {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.snapshotLocked()
}

// externalReject records a rejection produced by the Session layer (duplicate
// or expired detected from bounded session memory, or a structural refusal
// before the sample ever reached the aligner) into the same statistics the
// engine keeps.
func (e *Engine) externalReject(re *RejectError) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.recordReject(re)
}

// advanceInternal runs one deterministic advance step without the engine's
// public locking protocol; the Session holds its own mutex around it.
func (e *Engine) advanceInternal(target float64) (*model.State, []*RejectError, []*Measurement, error) {
	return e.advanceLocked(target)
}

// stateInternal returns a state copy without public locking.
func (e *Engine) stateInternal() *model.State { return e.f.State() }

// rejectionsInternal returns the raw rejection slice (caller must not mutate).
func (e *Engine) rejectionsInternal() []*RejectError { return e.rejected }
