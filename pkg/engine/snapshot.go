package engine

import (
	"errors"
	"fmt"
	"math"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/model"
)

// SnapshotFormatVersion is the engine snapshot layout version embedded in
// every archive. Loading an archive with a different version is refused.
const SnapshotFormatVersion = 1

// ErrSnapshotVersion indicates a snapshot was produced by an incompatible
// engine layout version and must not be restored.
var ErrSnapshotVersion = errors.New("engine: incompatible snapshot version")

// Snapshot is a complete, deep-copied, self-contained image of an Engine at a
// single point in time. Combined with the model registry it contains
// everything required to resume bit-identical processing in a new process.
type Snapshot struct {
	FormatVersion int
	Config        Config

	// EKF state.
	StateTime float64
	X         []float64
	P         []float64
	Dim       int
	Motion    *ModelRef

	// Aligner.
	AlignerNow float64
	Committed  map[string]float64
	// Pending holds buffered, not-yet-released measurements in their original
	// submission order. Processing order is recomputed by (time, source), so
	// the stored order never affects numerical results.
	Pending []PendingMeasurement

	// Health tracker.
	Health map[string]HealthSnapshot

	// Statistics and full rejection log (so resumed Stats/Rejections match an
	// uninterrupted run exactly).
	Stats      Stats
	Rejections []*RejectError
}

// ModelRef is a self-describing pointer to a model via its registered kind.
type ModelRef struct {
	Kind string
	Raw  []byte
}

// PendingMeasurement is a buffered, not-yet-released measurement.
type PendingMeasurement struct {
	Time      float64
	Source    string
	Z         []float64
	ModelKind string
	ModelRaw  []byte
}

// HealthSnapshot is the serialized fault state machine of one stream.
type HealthSnapshot struct {
	Faulty     bool
	BadStreak  int
	GoodStreak int
}

// Snapshot exports a consistent deep copy of the complete engine state. It
// takes the engine lock, so exports made while other goroutines keep
// submitting always represent one real point in time. An error is returned if
// any model in use does not implement model.Serializable.
func (e *Engine) Snapshot() (*Snapshot, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.snapshotLocked()
}

func (e *Engine) snapshotLocked() (*Snapshot, error) {
	kind, raw, err := model.EncodeMotion(e.f.motion)
	if err != nil {
		return nil, err
	}
	n := e.f.motion.Dim()
	s := &Snapshot{
		FormatVersion: SnapshotFormatVersion,
		Config:        e.cfg,
		StateTime:     e.f.state.Time,
		X:             append([]float64(nil), e.f.state.X.Data...),
		P:             append([]float64(nil), e.f.state.P.Data...),
		Dim:           n,
		Motion:        &ModelRef{Kind: kind, Raw: append([]byte(nil), raw...)},
		AlignerNow:    e.al.now,
		Committed:     make(map[string]float64, len(e.al.committed)),
		Pending:       make([]PendingMeasurement, 0, len(e.al.pending)),
		Health:        make(map[string]HealthSnapshot, len(e.ht.streams)),
	}
	for k, v := range e.al.committed {
		s.Committed[k] = v
	}
	for _, m := range e.al.pending {
		mk, mr, err := model.EncodeMeasurement(m.Model)
		if err != nil {
			return nil, fmt.Errorf("engine: pending measurement %s@%.6f: %w", m.Source, m.Time, err)
		}
		s.Pending = append(s.Pending, PendingMeasurement{
			Time:      m.Time,
			Source:    m.Source,
			Z:         append([]float64(nil), m.Z.Data...),
			ModelKind: mk,
			ModelRaw:  append([]byte(nil), mr...),
		})
	}
	for k, v := range e.ht.streams {
		s.Health[k] = HealthSnapshot{Faulty: v.faulty, BadStreak: v.badStreak, GoodStreak: v.goodStreak}
	}
	s.Stats = copyStats(e.stats)
	s.Rejections = make([]*RejectError, len(e.rejected))
	for i, r := range e.rejected {
		cp := *r
		s.Rejections[i] = &cp
	}
	return s, nil
}

func copyStats(in Stats) Stats {
	out := Stats{
		Predicts:  in.Predicts,
		Updates:   in.Updates,
		Rejected:  in.Rejected,
		PerStream: make(map[string]StreamStats, len(in.PerStream)),
	}
	for k, v := range in.PerStream {
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

// NewFromSnapshot rebuilds an engine from a validated snapshot. The caller is
// expected to have verified the envelope format/version and checksum; this
// function still validates the internal consistency of the snapshot and
// rejects it (returning an error) rather than ever running on half a state.
func NewFromSnapshot(s *Snapshot) (*Engine, error) {
	if s == nil {
		return nil, errors.New("engine: nil snapshot")
	}
	if s.FormatVersion != SnapshotFormatVersion {
		return nil, fmt.Errorf("%w: snapshot format version %d, want %d",
			ErrSnapshotVersion, s.FormatVersion, SnapshotFormatVersion)
	}
	if s.Motion == nil {
		return nil, errors.New("engine: snapshot missing motion model")
	}
	motion, err := model.DecodeMotion(s.Motion.Kind, s.Motion.Raw)
	if err != nil {
		return nil, fmt.Errorf("engine: restore motion model: %w", err)
	}
	n := motion.Dim()
	if s.Dim != n {
		return nil, fmt.Errorf("engine: snapshot dim %d != motion dim %d", s.Dim, n)
	}
	if len(s.X) != n {
		return nil, fmt.Errorf("engine: X length %d != dim %d", len(s.X), n)
	}
	if len(s.P) != n*n {
		return nil, fmt.Errorf("engine: P length %d != %d", len(s.P), n*n)
	}
	if math.IsNaN(s.StateTime) || math.IsInf(s.StateTime, 0) {
		return nil, errors.New("engine: snapshot state time not finite")
	}

	// Rebuild the state and let the EKF constructor validate finiteness /
	// symmetry / PSD exactly as it would for a fresh initial state.
	st := &model.State{
		Time: s.StateTime,
		X:    mat.AsVec(append([]float64(nil), s.X...)),
		P:    &mat.Matrix{Rows: n, Cols: n, Data: append([]float64(nil), s.P...)},
	}
	cfg := normalizeCfg(s.Config)
	f, err := NewEKF(st, motion, cfg.EigTol, cfg.CondLimit)
	if err != nil {
		return nil, fmt.Errorf("engine: restored state failed validation: %w", err)
	}
	// NewEKF cloned the state; the copy is already bit-identical.

	e := &Engine{
		cfg: cfg,
		f:   f,
		al:  newAligner(s.AlignerNow, cfg.MaxDelayHorizon),
		ht:  newHealthTracker(cfg.Health),
	}
	if math.IsNaN(s.AlignerNow) || math.IsInf(s.AlignerNow, 0) {
		return nil, errors.New("engine: snapshot aligner time not finite")
	}
	e.al.now = s.AlignerNow
	for src, t := range s.Committed {
		if math.IsNaN(t) || math.IsInf(t, 0) {
			return nil, errors.New("engine: snapshot committed time not finite")
		}
		e.al.committed[src] = t
	}
	for _, pm := range s.Pending {
		if math.IsNaN(pm.Time) || math.IsInf(pm.Time, 0) {
			return nil, errors.New("engine: snapshot pending time not finite")
		}
		md, err := model.DecodeMeasurement(pm.ModelKind, pm.ModelRaw)
		if err != nil {
			return nil, fmt.Errorf("engine: restore pending model %q: %w", pm.ModelKind, err)
		}
		if len(pm.Z) != md.Dim() {
			return nil, fmt.Errorf("engine: pending %s@%.6f z length %d != model dim %d",
				pm.Source, pm.Time, len(pm.Z), md.Dim())
		}
		e.al.pending = append(e.al.pending, &Measurement{
			Time:   pm.Time,
			Source: pm.Source,
			Z:      mat.AsVec(append([]float64(nil), pm.Z...)),
			Model:  md,
		})
	}
	for src, h := range s.Health {
		if h.BadStreak < 0 || h.GoodStreak < 0 {
			return nil, errors.New("engine: snapshot health counters negative")
		}
		e.ht.streams[src] = &streamHealth{faulty: h.Faulty, badStreak: h.BadStreak, goodStreak: h.GoodStreak}
	}
	e.stats = copyStats(s.Stats)
	e.rejected = make([]*RejectError, len(s.Rejections))
	for i, r := range s.Rejections {
		if r == nil {
			return nil, errors.New("engine: snapshot contains nil rejection entry")
		}
		cp := *r
		e.rejected[i] = &cp
	}
	if e.stats.Rejected != len(e.rejected) {
		return nil, fmt.Errorf("engine: snapshot inconsistent rejection count %d vs log %d",
			e.stats.Rejected, len(e.rejected))
	}
	return e, nil
}

// normalizeCfg applies the same default filling as New so a restored engine is
// configured identically.
func normalizeCfg(cfg Config) Config {
	d := DefaultConfig()
	if cfg.MaxDelayHorizon < 0 || math.IsNaN(cfg.MaxDelayHorizon) || math.IsInf(cfg.MaxDelayHorizon, 0) {
		cfg.MaxDelayHorizon = d.MaxDelayHorizon
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
	return cfg
}
