package engine

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/model"
)

// Archive format identification and compatibility.
const (
	// archiveMagic is the self-identifying prefix of every archive.
	archiveMagic = "SFES"
	// ArchiveFormatVersion is the only archive format version this build
	// writes and accepts.
	ArchiveFormatVersion = 1
)

// On-disk layout (all integers big-endian):
//
//	offset 0:  "SFES"                    (4 bytes magic)
//	offset 4:  format version uint16     (2 bytes)
//	offset 6:  payload length uint32     (4 bytes)
//	offset 10: payload JSON              (payloadLength bytes)
//	then:      sha256(payload)           (32 bytes)
//
// The payload carries its own engine-configuration, motion descriptor and
// per-source model descriptors, so an archive is self-describing. Floats are
// stored as IEEE-754 raw bits (uint64), so NaN/Inf round-trip exactly and no
// decimal re-parsing can change a single bit: a resumed run reproduces the
// uninterrupted run bit-for-bit.

// Sentinel errors reported when loading archives. Callers can use errors.Is to
// distinguish an unusable archive from any other failure: a session is never
// resumed from a half-read or incompatible archive.
var (
	// ErrArchiveCorrupted: truncated file, bad magic, checksum mismatch or
	// payload that cannot be decoded.
	ErrArchiveCorrupted = errors.New("archive corrupted")
	// ErrArchiveVersion: archive declares a format version this build cannot
	// read (older or newer).
	ErrArchiveVersion = errors.New("archive version not supported")
	// ErrArchiveBinding: the models/motion/config offered at restore time do
	// not match the descriptors recorded in the archive.
	ErrArchiveBinding = errors.New("archive binding mismatch")
)

const (
	headerLen = 10 // magic(4) + version(2) + payload length(4)
	sumLen    = sha256.Size
)

func fbits(f float64) uint64 { return math.Float64bits(f) }
func funit(b uint64) float64 { return math.Float64frombits(b) }

func vecBits(v *mat.Vector) []uint64 {
	if v == nil {
		return nil
	}
	out := make([]uint64, len(v.Data))
	for i, x := range v.Data {
		out[i] = math.Float64bits(x)
	}
	return out
}

func bitsVec(bits []uint64) *mat.Vector {
	v := mat.New(len(bits), 1)
	for i, b := range bits {
		v.Data[i] = math.Float64frombits(b)
	}
	return v
}

func reasonKeys(m map[RejectReason]int) map[string]int {
	if m == nil {
		return nil
	}
	out := make(map[string]int, len(m))
	for k, v := range m {
		out[string(k)] = v
	}
	return out
}

func keysReason(m map[string]int) map[RejectReason]int {
	if m == nil {
		return nil
	}
	out := make(map[RejectReason]int, len(m))
	for k, v := range m {
		out[RejectReason(k)] = v
	}
	return out
}

// ---- wire payload -------------------------------------------------------

type wireHealth struct {
	Source string
	Faulty bool
	Bad    int
	Good   int
}

type wireStreamStats struct {
	Source        string
	Accepted      int
	HardFaults    int
	SoftFaults    int
	Recoveries    int
	CurrentlyDown bool
	Rejected      map[string]int
}

type wirePending struct {
	TimeBits uint64
	Source   string
	Z        []uint64
}

type wireReject struct {
	Source string
	Time   uint64
	Reason string
	Detail string
}

type wireEngine struct {
	MaxDelayHorizon uint64
	EigTol          uint64
	CondLimit       uint64
	GateSigma       uint64
	FaultStreak     int
	RecoverStreak   int

	StateTime uint64
	StateDim  int
	X         []uint64
	PRowMajor []uint64

	AlignerNow uint64
	Pending    []wirePending
	Committed  map[string]uint64
	Health     []wireHealth

	Predicts  int
	Updates   int
	RejectedN int
	PerStream []wireStreamStats
	Rejects   []wireReject
}

type wireOp struct {
	Kind     int // 0 submit, 1 advance
	Source   string
	TimeBits uint64
	Target   uint64
	Z        []uint64
}

type wireSeen struct {
	Source string
	Time   uint64
}

type wirePayload struct {
	Version int

	MotionDescriptor string
	Bindings         []wireBinding

	MemoryLimit int
	Clock       uint64
	Engine      wireEngine

	CP   wireEngine
	Ops  []wireOp
	Seen []wireSeen
}

type wireBinding struct {
	Source     string
	Descriptor string
}

// ---- encoding -----------------------------------------------------------

func engineToWire(sn *engineSnapshot) wireEngine {
	w := wireEngine{
		MaxDelayHorizon: fbits(sn.CFG.MaxDelayHorizon),
		EigTol:          fbits(sn.CFG.EigTol),
		CondLimit:       fbits(sn.CFG.CondLimit),
		GateSigma:       fbits(sn.CFG.Health.GateSigma),
		FaultStreak:     sn.CFG.Health.FaultStreak,
		RecoverStreak:   sn.CFG.Health.RecoverStreak,
		StateTime:       fbits(sn.State.Time),
		StateDim:        sn.State.X.Rows,
		X:               vecBits(sn.State.X),
		PRowMajor:       vecBits(&mat.Vector{Rows: sn.State.P.Rows * sn.State.P.Cols, Cols: 1, Data: sn.State.P.Data}),
		AlignerNow:      fbits(sn.AlignerNow),
		Committed:       make(map[string]uint64, len(sn.Committed)),
		Health:          make([]wireHealth, 0, len(sn.Health)),
		Predicts:        sn.Stats.Predicts,
		Updates:         sn.Stats.Updates,
		RejectedN:       sn.Stats.Rejected,
		PerStream:       make([]wireStreamStats, 0, len(sn.Stats.PerStream)),
		Rejects:         make([]wireReject, 0, len(sn.Rejected)),
	}
	for _, p := range sn.Pending {
		w.Pending = append(w.Pending, wirePending{TimeBits: fbits(p.Time), Source: p.Source, Z: vecBits(p.Z)})
	}
	for k, v := range sn.Committed {
		w.Committed[k] = fbits(v)
	}
	for k, v := range sn.Health {
		w.Health = append(w.Health, wireHealth{Source: k, Faulty: v.Faulty, Bad: v.BadStreak, Good: v.GoodStreak})
	}
	for k, v := range sn.Stats.PerStream {
		w.PerStream = append(w.PerStream, wireStreamStats{
			Source: k, Accepted: v.Accepted, HardFaults: v.HardFaults,
			SoftFaults: v.SoftFaults, Recoveries: v.Recoveries,
			CurrentlyDown: v.CurrentlyDown, Rejected: reasonKeys(v.Rejected),
		})
	}
	for _, r := range sn.Rejected {
		w.Rejects = append(w.Rejects, wireReject{
			Source: r.Source, Time: fbits(r.Time), Reason: string(r.Reason), Detail: r.Detail,
		})
	}
	return w
}

func opToWire(op logOp) wireOp {
	w := wireOp{Kind: int(op.kind)}
	if op.kind == opSubmit {
		w.Source = op.source
		w.TimeBits = fbits(op.time)
		w.Z = vecBits(op.z)
	} else {
		w.Target = fbits(op.target)
	}
	return w
}

// encodeLocked must be called with s.mu held.
func (s *Session) encodeLocked() ([]byte, error) {
	p := wirePayload{
		Version:          ArchiveFormatVersion,
		MotionDescriptor: s.motionDesc,
		MemoryLimit:      s.cfg.MemoryLimit,
		Clock:            fbits(s.clock),
		Engine:           engineToWire(s.eng.snapshotInternal()),
		CP:               engineToWire(s.cp),
		Ops:              make([]wireOp, 0, len(s.ops)),
		Seen:             make([]wireSeen, 0, len(s.seen)),
	}
	for _, b := range sortedBindings(s.descs) {
		p.Bindings = append(p.Bindings, wireBinding{Source: b.source, Descriptor: b.desc})
	}
	for _, op := range s.ops {
		p.Ops = append(p.Ops, opToWire(op))
	}
	for k := range s.seen {
		p.Seen = append(p.Seen, wireSeen{Source: k.source, Time: fbits(k.time)})
	}
	payload, err := json.Marshal(&p)
	if err != nil {
		return nil, fmt.Errorf("%w: cannot encode payload: %v", ErrArchiveCorrupted, err)
	}
	if len(payload) > math.MaxUint32 {
		return nil, fmt.Errorf("%w: payload too large", ErrArchiveCorrupted)
	}
	out := make([]byte, 0, headerLen+len(payload)+sumLen)
	out = append(out, archiveMagic...)
	var hv [2]byte
	binary.BigEndian.PutUint16(hv[:], uint16(ArchiveFormatVersion))
	out = append(out, hv[:]...)
	var lv [4]byte
	binary.BigEndian.PutUint32(lv[:], uint32(len(payload)))
	out = append(out, lv[:]...)
	out = append(out, payload...)
	sum := sha256.Sum256(payload)
	out = append(out, sum[:]...)
	return out, nil
}

type bindingPair struct {
	source, desc string
}

func sortedBindings(descs map[string]string) []bindingPair {
	out := make([]bindingPair, 0, len(descs))
	for k, v := range descs {
		out = append(out, bindingPair{k, v})
	}
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j].source < out[j-1].source; j-- {
			out[j-1], out[j] = out[j], out[j-1]
		}
	}
	return out
}

// ---- decoding -----------------------------------------------------------

func engineFromWire(w wireEngine) (*engineSnapshot, error) {
	n := w.StateDim
	if n <= 0 || len(w.X) != n || len(w.PRowMajor) != n*n {
		return nil, fmt.Errorf("%w: state dimensions inconsistent", ErrArchiveCorrupted)
	}
	sn := &engineSnapshot{
		CFG: Config{
			MaxDelayHorizon: funit(w.MaxDelayHorizon),
			EigTol:          funit(w.EigTol),
			CondLimit:       funit(w.CondLimit),
			Health: HealthConfig{
				GateSigma: funit(w.GateSigma), FaultStreak: w.FaultStreak, RecoverStreak: w.RecoverStreak,
			},
		},
		State: &model.State{
			Time: funit(w.StateTime),
			X:    bitsVec(w.X),
			P:    &mat.Matrix{Rows: n, Cols: n, Data: append([]float64(nil), floatsOf(w.PRowMajor)...)},
		},
		AlignerNow: funit(w.AlignerNow),
		Pending:    make([]pendingMeasurement, 0, len(w.Pending)),
		Committed:  make(map[string]float64, len(w.Committed)),
		Health:     make(map[string]streamHealthData, len(w.Health)),
		Stats: Stats{
			Predicts:  w.Predicts,
			Updates:   w.Updates,
			Rejected:  w.RejectedN,
			PerStream: make(map[string]StreamStats, len(w.PerStream)),
		},
		Rejected: make([]rejectionData, 0, len(w.Rejects)),
	}
	for _, p := range w.Pending {
		sn.Pending = append(sn.Pending, pendingMeasurement{Time: funit(p.TimeBits), Source: p.Source, Z: bitsVec(p.Z)})
	}
	for k, v := range w.Committed {
		sn.Committed[k] = funit(v)
	}
	for _, h := range w.Health {
		sn.Health[h.Source] = streamHealthData{Faulty: h.Faulty, BadStreak: h.Bad, GoodStreak: h.Good}
	}
	for _, st := range w.PerStream {
		sn.Stats.PerStream[st.Source] = StreamStats{
			Accepted: st.Accepted, HardFaults: st.HardFaults, SoftFaults: st.SoftFaults,
			Recoveries: st.Recoveries, CurrentlyDown: st.CurrentlyDown, Rejected: keysReason(st.Rejected),
		}
	}
	for _, r := range w.Rejects {
		sn.Rejected = append(sn.Rejected, rejectionData{
			Source: r.Source, Time: funit(r.Time), Reason: RejectReason(r.Reason), Detail: r.Detail,
		})
	}
	return sn, nil
}

func floatsOf(bits []uint64) []float64 {
	out := make([]float64, len(bits))
	for i, b := range bits {
		out[i] = math.Float64frombits(b)
	}
	return out
}

func restoreSessionBytes(data []byte, motion model.MotionModel, motionDescriptor string, bindings []SourceBinding, cfg SessionConfig) (*Session, error) {
	if motion == nil {
		return nil, fmt.Errorf("%w: nil motion model", ErrArchiveBinding)
	}
	if len(data) < headerLen+sumLen {
		return nil, fmt.Errorf("%w: archive shorter than header+checksum", ErrArchiveCorrupted)
	}
	if string(data[:4]) != archiveMagic {
		return nil, fmt.Errorf("%w: bad magic %q", ErrArchiveCorrupted, string(data[:4]))
	}
	version := binary.BigEndian.Uint16(data[4:6])
	if int(version) != ArchiveFormatVersion {
		return nil, fmt.Errorf("%w: archive version=%d, this build supports version %d",
			ErrArchiveVersion, version, ArchiveFormatVersion)
	}
	plen := int(binary.BigEndian.Uint32(data[6:10]))
	if plen != len(data)-headerLen-sumLen {
		return nil, fmt.Errorf("%w: declared payload length %d does not match file size %d",
			ErrArchiveCorrupted, plen, len(data)-headerLen-sumLen)
	}
	payload := data[headerLen : headerLen+plen]
	storedSum := data[headerLen+plen:]
	want := sha256.Sum256(payload)
	for i := range want {
		if want[i] != storedSum[i] {
			return nil, fmt.Errorf("%w: sha256 checksum mismatch", ErrArchiveCorrupted)
		}
	}

	var p wirePayload
	if err := json.Unmarshal(payload, &p); err != nil {
		return nil, fmt.Errorf("%w: payload not decodable: %v", ErrArchiveCorrupted, err)
	}
	if p.Version != ArchiveFormatVersion {
		return nil, fmt.Errorf("%w: payload version=%d", ErrArchiveVersion, p.Version)
	}
	if motionDescriptor != p.MotionDescriptor {
		return nil, fmt.Errorf("%w: motion descriptor mismatch (archived=%q provided=%q)",
			ErrArchiveBinding, p.MotionDescriptor, motionDescriptor)
	}

	// Binding verification: same source set, same descriptors; motion
	// descriptor and exact engine configuration must match, otherwise the
	// continuation cannot be bit-identical.
	models := make(map[string]model.MeasurementModel, len(bindings))
	descs := make(map[string]string, len(bindings))
	for _, b := range bindings {
		if b.Source == "" || b.Model == nil {
			return nil, fmt.Errorf("%w: empty source or nil model in bindings", ErrArchiveBinding)
		}
		if _, dup := models[b.Source]; dup {
			return nil, fmt.Errorf("%w: duplicate binding for source %s", ErrArchiveBinding, b.Source)
		}
		models[b.Source] = b.Model
		descs[b.Source] = b.Descriptor
	}
	if len(descs) != len(p.Bindings) {
		return nil, fmt.Errorf("%w: binding set size %d != archived %d",
			ErrArchiveBinding, len(descs), len(p.Bindings))
	}
	for _, ab := range p.Bindings {
		d, ok := descs[ab.Source]
		if !ok || d != ab.Descriptor {
			return nil, fmt.Errorf("%w: source %q descriptor mismatch (archived=%q provided=%q)",
				ErrArchiveBinding, ab.Source, ab.Descriptor, d)
		}
	}
	if cfg.MemoryLimit <= 0 {
		cfg.MemoryLimit = p.MemoryLimit
	}
	archivedCfg := configFromWire(p.Engine)
	if cfg.Engine != archivedCfg {
		return nil, fmt.Errorf("%w: engine config differs from the one recorded in the archive", ErrArchiveBinding)
	}

	engSN, err := engineFromWire(p.Engine)
	if err != nil {
		return nil, err
	}
	cpSN, err := engineFromWire(p.CP)
	if err != nil {
		return nil, err
	}
	eng, err := restoreEngine(engSN, motion, models)
	if err != nil {
		return nil, err
	}
	// Sanity: the recorded checkpoint must itself be restorable under the
	// supplied motion model and bindings.
	if _, err := restoreEngine(cpSN, motion, models); err != nil {
		return nil, err
	}

	s := &Session{
		cfg:        cfg,
		motion:     motion,
		motionDesc: p.MotionDescriptor,
		models:     models,
		descs:      descs,
		eng:        eng,
		clock:      funit(p.Clock),
		cp:         cpSN,
		ops:        make([]logOp, 0, len(p.Ops)),
		seen:       make(map[dedupKey]struct{}, len(p.Seen)),
	}
	for _, op := range p.Ops {
		switch op.Kind {
		case int(opSubmit):
			s.ops = append(s.ops, logOp{kind: opSubmit, source: op.Source, time: funit(op.TimeBits), z: bitsVec(op.Z)})
		case int(opAdvance):
			s.ops = append(s.ops, logOp{kind: opAdvance, target: funit(op.Target)})
		default:
			return nil, fmt.Errorf("%w: unknown op kind %d", ErrArchiveCorrupted, op.Kind)
		}
	}
	for _, k := range p.Seen {
		s.seen[dedupKey{k.Source, funit(k.Time)}] = struct{}{}
	}
	return s, nil
}

func configFromWire(w wireEngine) Config {
	return Config{
		MaxDelayHorizon: funit(w.MaxDelayHorizon),
		EigTol:          funit(w.EigTol),
		CondLimit:       funit(w.CondLimit),
		Health: HealthConfig{
			GateSigma: funit(w.GateSigma), FaultStreak: w.FaultStreak, RecoverStreak: w.RecoverStreak,
		},
	}
}
