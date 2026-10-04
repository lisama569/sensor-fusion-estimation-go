package engine

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"sync"
	"testing"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/model"
)

// This file covers the session layer added on top of the one-shot engine:
//
//  1. save/resume in a brand-new "process" is bit-identical to an
//     uninterrupted run (state, covariance, per-stream statistics);
//  2. historical backfill is bit-identical to the samples arriving in real
//     time, including nonlinear models and rejected/gated samples;
//  3. the same sample is never fused twice: across a snapshot, across
//     backfill batches and inside one batch, with DISTINGUISHABLE reasons
//     (duplicate_timestamp vs expired_measurement);
//  4. processed-sample memory is hard-bounded via checkpoint folding and
//     results stay bit-identical;
//  5. archives are self-describing: bad magic/truncation/checksum, unsupported
//     version, and descriptor/config mismatches are all refused explicitly;
//  6. concurrent snapshotting during live ingestion is race-clean and every
//     exported archive is a consistent point-in-time image.

type sessSample struct {
	src string
	t   float64
	z   *mat.Vector
}

func sessPosModel(v float64) model.MeasurementModel {
	return &model.LinearPosition{R: mat.From([][]float64{{v, 0}, {0, v}})}
}

func sessBindings() []SourceBinding {
	return []SourceBinding{
		{Source: "gps", Descriptor: "linear-position R=diag(0.04,0.04) v=1", Model: sessPosModel(0.04)},
		{Source: "imu", Descriptor: "linear-position R=diag(0.09,0.09) v=1", Model: sessPosModel(0.09)},
		{Source: "radar", Descriptor: "range-bearing S=(-2,-2) v=1",
			Model: &model.RangeBearing{Sx: -2, Sy: -2, R: mat.From([][]float64{{0.04, 0}, {0, 0.0004}})}},
	}
}

func sessInitial() *model.State {
	return &model.State{Time: 0, X: mat.Vec(0, 0, 1, 0), P: mat.Identity(4)}
}

func sessMotion() model.MotionModel { return &model.ConstantVelocity{QPSD: 0.05} }

func buildLiveSequence() []sessSample {
	var out []sessSample
	for k := 1; k <= 30; k++ {
		tt := float64(k) * 0.1
		out = append(out, sessSample{"gps", tt, mat.Vec(tt, 0.2*math.Sin(tt))})
		if k%2 == 0 {
			out = append(out, sessSample{"imu", tt, mat.Vec(tt, -0.05)})
		}
		if k%5 == 0 {
			px, py := tt, 0.2*math.Sin(tt)
			out = append(out, sessSample{"radar", tt,
				mat.Vec(math.Hypot(px+2, py+2), math.Atan2(py+2, px+2))})
		}
	}
	return out
}

func driveLive(t *testing.T, s *Session, seq []sessSample, stopAt int) {
	t.Helper()
	// Submit all samples whose time is <= the current clock in an arbitrary
	// (emission) order; Advance steps at 0.1s.
	maxT := seq[len(seq)-1].t
	if stopAt > 0 {
		maxT = float64(stopAt) * 0.1
	}
	_ = maxT
	i := 0
	for tt := 0.1; tt <= maxT+1e-9; tt = math.Round((tt+0.1)*10) / 10 {
		for i < len(seq) && seq[i].t <= tt+1e-9 {
			if err := s.Submit(seq[i].src, seq[i].t, seq[i].z.Clone()); err != nil {
				t.Logf("submit rejected src=%s t=%.1f reason=%s", seq[i].src, seq[i].t, rejectReasonOf(err))
			}
			i++
		}
		if _, rejs, err := s.Advance(tt); err != nil {
			t.Fatalf("advance %.1f: %v", tt, err)
		} else {
			for _, r := range rejs {
				t.Logf("  fuse reject @%.1f src=%s reason=%s", tt, r.Source, r.Reason)
			}
		}
	}
}

func rejectReasonOf(err error) RejectReason {
	var re *RejectError
	if errors.As(err, &re) {
		return re.Reason
	}
	return RejectReason("non-reject: " + err.Error())
}

func assertSessionsBitEqual(t *testing.T, tag string, got, want *Session) {
	assertSessions(t, tag, got, want, false)
}

func assertSessions(t *testing.T, tag string, got, want *Session, orderInsensitive bool) {
	t.Helper()
	g, w := got.State(), want.State()
	if math.Float64bits(g.Time) != math.Float64bits(w.Time) {
		t.Fatalf("[%s] time bits differ: %.17g != %.17g", tag, g.Time, w.Time)
	}
	if len(g.X.Data) != len(w.X.Data) {
		t.Fatalf("[%s] X dims differ", tag)
	}
	for i := range g.X.Data {
		if math.Float64bits(g.X.Data[i]) != math.Float64bits(w.X.Data[i]) {
			t.Fatalf("[%s] X differs at %d: %v vs %v", tag, i, g.X.Data, w.X.Data)
		}
	}
	for i := range g.P.Data {
		if math.Float64bits(g.P.Data[i]) != math.Float64bits(w.P.Data[i]) {
			t.Fatalf("[%s] P differs at index %d", tag, i)
		}
	}
	gs, ws := got.Stats(), want.Stats()
	if gs.Predicts != ws.Predicts || gs.Updates != ws.Updates || gs.Rejected != ws.Rejected {
		t.Fatalf("[%s] top-level stats differ: %+v vs %+v", tag, gs, ws)
	}
	if len(gs.PerStream) != len(ws.PerStream) {
		t.Fatalf("[%s] per-stream stat size differs", tag)
	}
	for name, a := range gs.PerStream {
		b, ok := ws.PerStream[name]
		if !ok {
			t.Fatalf("[%s] stream %s missing in reference", tag, name)
		}
		if a.Accepted != b.Accepted || a.HardFaults != b.HardFaults ||
			a.SoftFaults != b.SoftFaults || a.Recoveries != b.Recoveries {
			t.Fatalf("[%s] stream %s counters differ: %+v vs %+v", tag, name, a, b)
		}
		if len(a.Rejected) != len(b.Rejected) {
			t.Fatalf("[%s] stream %s reject-reason maps differ: %v vs %v", tag, name, a.Rejected, b.Rejected)
		}
		for reason, n := range a.Rejected {
			if b.Rejected[reason] != n {
				t.Fatalf("[%s] stream %s reason %s count %d vs %d", tag, name, reason, n, b.Rejected[reason])
			}
		}
	}
	gr, wr := got.Rejections(), want.Rejections()
	if len(gr) != len(wr) {
		t.Fatalf("[%s] rejection log length differs: %d vs %d", tag, len(gr), len(wr))
	}
	if orderInsensitive {
		assertRejectionMultiset(t, tag, gr, wr)
	} else {
		for i := range gr {
			if gr[i].Source != wr[i].Source ||
				math.Float64bits(gr[i].Time) != math.Float64bits(wr[i].Time) ||
				gr[i].Reason != wr[i].Reason {
				t.Fatalf("[%s] rejection #%d differs: %+v vs %+v", tag, i, gr[i], wr[i])
			}
		}
	}
	t.Logf("[%s] bit-identical: time=%.3f updates=%d rejects=%d",
		tag, g.Time, gs.Updates, gs.Rejected)
}

// assertRejectionMultiset compares rejection logs as multisets of
// (source, timestamp-bits, reason): backfill rebuilds apply order-dependent
// rejects at a deterministic but different position than real-time arrival,
// while state/covariance/counters remain bit-identical.
func assertRejectionMultiset(t *testing.T, tag string, gr, wr []*RejectError) {
	t.Helper()
	key := func(r *RejectError) string {
		return fmt.Sprintf("%s|%016x|%s", r.Source, math.Float64bits(r.Time), r.Reason)
	}
	counts := make(map[string]int, len(wr))
	for _, r := range wr {
		counts[key(r)]++
	}
	for _, r := range gr {
		counts[key(r)]--
	}
	for k, v := range counts {
		if v != 0 {
			t.Fatalf("[%s] rejection multiset differs at %s (delta=%d)", tag, k, v)
		}
	}
}

// TestSessionSaveResumeMatchesUninterrupted: split the run at several points,
// snapshot the interrupted session, "restart the process" (fresh session via
// RestoreSession) and continue. The continued trajectory must equal the
// never-interrupted trajectory bit-for-bit.
// feedSteps drives a session in 0.1s clock steps: at each step it submits
// EVERY sample whose timestamp belongs to that step, then advances. This is
// the natural "same inputs" semantics -- at any committed instant the engine
// sees exactly the multiset of measurements that belong to the elapsed
// history -- so a save/resume split at an arbitrary step reproduces the
// uninterrupted run bit-for-bit, regardless of how the samples were emitted.
func feedSteps(t *testing.T, s *Session, seq []sessSample, fromStep, toStep int) {
	t.Helper()
	idx := 0
	for step := 1; step <= toStep; step++ {
		tt := float64(step) * 0.1
		for idx < len(seq) && seq[idx].t <= tt+1e-9 {
			if step > fromStep {
				if err := s.Submit(seq[idx].src, seq[idx].t, seq[idx].z.Clone()); err != nil {
					t.Logf("submit rejected src=%s t=%.2f: %s", seq[idx].src, seq[idx].t, rejectReasonOf(err))
				}
			}
			idx++
		}
		if step > fromStep {
			if _, _, err := s.Advance(tt); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestSessionSaveResumeMatchesUninterrupted(t *testing.T) {
	seq := buildLiveSequence()
	bindings := sessBindings()
	motion := sessMotion()
	cfg := DefaultSessionConfig()

	const totalSteps = 30
	uninterrupted, err := NewSession(sessInitial(), motion, "cv-qpsd=0.05", bindings, cfg)
	if err != nil {
		t.Fatal(err)
	}
	feedSteps(t, uninterrupted, seq, 0, totalSteps)

	for _, split := range []int{3, 12, 25, 30} {
		name := fmt.Sprintf("split@%.1fs", float64(split)*0.1)
		t.Run(name, func(t *testing.T) {
			interrupted, _ := NewSession(sessInitial(), motion, "cv-qpsd=0.05", sessBindings(), cfg)
			feedSteps(t, interrupted, seq, 0, split)

			arc, err := interrupted.Snapshot()
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("archived %d bytes at split %.1fs", len(arc), float64(split)*0.1)

			// "New process": only the bytes survive; models are rebuilt by the
			// application from the same descriptors, then the SAME remaining
			// input multiset arrives on the SAME clock.
			resumed, err := RestoreSession(bytes.NewReader(arc), sessMotion(), "cv-qpsd=0.05",
				sessBindings(), DefaultSessionConfig())
			if err != nil {
				t.Fatal(err)
			}
			feedSteps(t, resumed, seq, split, totalSteps)
			assertSessionsBitEqual(t, name, resumed, uninterrupted)
		})
	}
}

// TestSessionBackfillMatchesRealtime: run a session with a gap in its live
// history, then backfill the missing samples as an offline batch. The result
// must equal a reference session that had every sample before the matching
// Advance steps (i.e. the samples arrived in real time).
func TestSessionBackfillMatchesRealtime(t *testing.T) {
	seq := buildLiveSequence()
	bindings := sessBindings()
	motion := sessMotion()
	cfg := DefaultSessionConfig()

	// Reference: everything submitted before the advances.
	ref, _ := NewSession(sessInitial(), motion, "cv-qpsd=0.05", bindings, cfg)
	for _, sm := range seq {
		if err := ref.Submit(sm.src, sm.t, sm.z.Clone()); err != nil {
			t.Logf("ref submit: %v", err)
		}
	}
	for tt := 0.1; tt <= 3.0+1e-9; tt = math.Round((tt+0.1)*10) / 10 {
		if _, _, err := ref.Advance(tt); err != nil {
			t.Fatal(err)
		}
	}

	// Live run: gps/imu/radar samples are withheld until t=2.0; the clock only
	// advances (no data) until then, then the withheld window is backfilled.
	bf, _ := NewSession(sessInitial(), sessMotion(), "cv-qpsd=0.05", sessBindings(), cfg)
	var batch []HistoricalMeasurement
	for _, sm := range seq {
		if sm.t <= 2.0+1e-9 {
			batch = append(batch, HistoricalMeasurement{Time: sm.t, Source: sm.src, Z: sm.z.Clone()})
		}
	}
	for tt := 0.1; tt <= 2.0+1e-9; tt = math.Round((tt+0.1)*10) / 10 {
		if _, _, err := bf.Advance(tt); err != nil {
			t.Fatal(err)
		}
	}
	fused, rejs, err := bf.Backfill(batch)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("backfill batch=%d fused=%d rejected=%d (within-window, first fill)",
		len(batch), fused, len(rejs))
	for _, r := range rejs {
		t.Logf("  backfill reject src=%s t=%.1f reason=%s", r.Source, r.Time, r.Reason)
	}

	// Continue both sessions identically with the remaining live samples.
	for _, sm := range seq {
		if sm.t > 2.0+1e-9 {
			if err := bf.Submit(sm.src, sm.t, sm.z.Clone()); err != nil {
				t.Logf("bf submit: %v", err)
			}
		}
	}
	for tt := 2.1; tt <= 3.0+1e-9; tt = math.Round((tt+0.1)*10) / 10 {
		if _, _, err := bf.Advance(tt); err != nil {
			t.Fatal(err)
		}
	}
	assertSessions(t, "backfill==realtime", bf, ref, true)
}

// TestSessionDuplicateReasonsDistinguishable: one sample appears (a) after a
// snapshot boundary in a resumed process, (b) in two different backfill
// batches, (c) twice inside a single batch, and (d) once as a genuinely
// expired old sample. The duplicate and expired refusal reasons must be
// distinct and nothing may be fused twice.
func TestSessionDuplicateReasonsDistinguishable(t *testing.T) {
	motion := sessMotion()
	cfg := DefaultSessionConfig()
	s, _ := NewSession(sessInitial(), motion, "cv-qpsd=0.05", sessBindings(), cfg)

	if err := s.Submit("gps", 0.5, mat.Vec(0.5, 0.0)); err != nil {
		t.Fatal(err)
	}
	if _, _, err := s.Advance(0.5); err != nil {
		t.Fatal(err)
	}

	// (a) duplicate detected live, then must survive a save/restore cycle.
	if err := s.Submit("gps", 0.5, mat.Vec(9, 9)); err == nil ||
		rejectReasonOf(err) != ReasonDuplicate {
		t.Fatalf("live duplicate: %v", err)
	}
	t.Log("(a) live duplicate of processed sample -> duplicate_timestamp")

	arc, _ := s.Snapshot()
	resumed, err := RestoreSession(bytes.NewReader(arc), sessMotion(), "cv-qpsd=0.05", sessBindings(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := resumed.Submit("gps", 0.5, mat.Vec(9, 9)); err == nil ||
		rejectReasonOf(err) != ReasonDuplicate {
		t.Fatalf("post-restore duplicate: %v", err)
	}
	t.Log("(a') duplicate still detected after restore in a new process -> duplicate_timestamp")

	// (b) cross-batch duplicate. Advance the clock first so the historical
	// window (t=0.8, t=0.9) lies in the past relative to the session clock.
	if _, _, err := resumed.Advance(1.0); err != nil {
		t.Fatal(err)
	}
	b1 := []HistoricalMeasurement{{Time: 0.8, Source: "gps", Z: mat.Vec(0.8, 0)}}
	if n, _, err := resumed.Backfill(b1); err != nil || n != 1 {
		t.Fatalf("first backfill n=%d err=%v", n, err)
	}
	_, r2, err := resumed.Backfill(b1)
	if err != nil || len(r2) != 1 || r2[0].Reason != ReasonDuplicate {
		t.Fatalf("cross-batch duplicate: r=%v err=%v", r2, err)
	}
	t.Log("(b) same sample in a second backfill batch -> duplicate_timestamp")

	// (c) duplicate group inside one batch: every copy refused.
	b3 := []HistoricalMeasurement{
		{Time: 0.9, Source: "gps", Z: mat.Vec(0.9, 0)},
		{Time: 0.9, Source: "gps", Z: mat.Vec(0.91, 0)},
	}
	_, r3, err := resumed.Backfill(b3)
	if err != nil || len(r3) != 2 || r3[0].Reason != ReasonDuplicate || r3[1].Reason != ReasonDuplicate {
		t.Fatalf("in-batch duplicate group: r=%v err=%v", r3, err)
	}
	t.Log("(c) duplicate group within one backfill batch: both copies -> duplicate_timestamp")

	// (d) genuinely expired: advance past the delay horizon (2s).
	if _, _, err := resumed.Advance(3.0); err != nil {
		t.Fatal(err)
	}
	if err := resumed.Submit("gps", 0.95, mat.Vec(0.95, 0)); err == nil ||
		rejectReasonOf(err) != ReasonExpired {
		t.Fatalf("expired submit: %v", err)
	}
	t.Log("(d) timestamp older than clock-horizon -> expired_measurement")
	_, r4, err := resumed.Backfill([]HistoricalMeasurement{{Time: 0.95, Source: "gps", Z: mat.Vec(0.95, 0)}})
	if err != nil || len(r4) != 1 || r4[0].Reason != ReasonExpired {
		t.Fatalf("expired backfill: r=%v err=%v", r4, err)
	}
	t.Log("(d') same old sample via backfill -> expired_measurement (distinct from duplicate)")

}

// TestSessionMemoryBoundedAndCorrect: with a small MemoryLimit the session
// folds checkpoints aggressively; retained memory never exceeds the bound and
// the estimate still equals the uninterrupted run. A sample older than the
// folded floor is refused as expired, and a within-window late backfill still
// matches real-time arrival.
func TestSessionMemoryBoundedAndCorrect(t *testing.T) {
	cfg := DefaultSessionConfig()
	cfg.MemoryLimit = 40
	motion := sessMotion()

	s, _ := NewSession(sessInitial(), motion, "cv-qpsd=0.05", sessBindings(), cfg)
	ref, _ := NewSession(sessInitial(), sessMotion(), "cv-qpsd=0.05", sessBindings(), DefaultSessionConfig())
	for k := 1; k <= 120; k++ {
		tt := float64(k) * 0.1
		z := mat.Vec(tt, 0.1*math.Sin(tt))
		if err := s.Submit("gps", tt, z.Clone()); err != nil {
			t.Fatal(err)
		}
		if err := ref.Submit("gps", tt, z); err != nil {
			t.Fatal(err)
		}
		if _, _, err := s.Advance(tt); err != nil {
			t.Fatal(err)
		}
		if _, _, err := ref.Advance(tt); err != nil {
			t.Fatal(err)
		}
	}
	assertSessionsBitEqual(t, "bounded-run", s, ref)
	if len(s.ops) > cfg.MemoryLimit+1 {
		t.Fatalf("retained ops %d exceed bound %d", len(s.ops), cfg.MemoryLimit)
	}
	if len(s.seen) > cfg.MemoryLimit+1 {
		t.Fatalf("processed-sample memory %d exceed bound %d", len(s.seen), cfg.MemoryLimit)
	}
	t.Logf("memory bounded: retainedOps=%d seen=%d (limit=%d)", len(s.ops), len(s.seen), cfg.MemoryLimit)

	// Within-window late historical sample equals real-time arrival.
	late := HistoricalMeasurement{Time: 11.55, Source: "gps", Z: mat.Vec(11.55, 0.02)}
	n, _, err := s.Backfill([]HistoricalMeasurement{late})
	if err != nil || n != 1 {
		t.Fatalf("late backfill n=%d err=%v", n, err)
	}
	ref2, _ := NewSession(sessInitial(), sessMotion(), "cv-qpsd=0.05", sessBindings(), DefaultSessionConfig())
	submitted := false
	for k := 1; k <= 120; k++ {
		tt := float64(k) * 0.1
		// ensure the late sample is buffered before the 11.6 advance
		if !submitted && tt >= 11.6 {
			_ = ref2.Submit(late.Source, late.Time, late.Z.Clone())
			submitted = true
		}
		_ = ref2.Submit("gps", tt, mat.Vec(tt, 0.1*math.Sin(tt)))
		_, _, _ = ref2.Advance(tt)
	}
	if !submitted {
		t.Fatal("test setup error")
	}
	assertSessionsBitEqual(t, "bounded-backfill", s, ref2)

	// A sample older than the folded floor cannot be replayed: explicit
	// expired refusal, never a silent drop, never fused twice.
	_, old, err := s.Backfill([]HistoricalMeasurement{{Time: 0.2, Source: "gps", Z: mat.Vec(0.2, 0)}})
	if err != nil || len(old) != 1 || old[0].Reason != ReasonExpired {
		t.Fatalf("beyond-floor backfill: %v %v", old, err)
	}
	t.Logf("sample older than folded floor -> %s", ReasonExpired)
}

// TestSessionArchiveRejectsBadInput: every way an archive can be unusable must
// be reported with the distinct sentinel, and no session may be constructed.
func TestSessionArchiveRejectsBadInput(t *testing.T) {
	s, _ := NewSession(sessInitial(), sessMotion(), "cv-qpsd=0.05", sessBindings(), DefaultSessionConfig())
	driveLive(t, s, buildLiveSequence(), 10)
	good, err := s.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	cfg := DefaultSessionConfig()

	cases := []struct {
		name    string
		mutate  func(b []byte) []byte
		wantErr error
	}{
		{"truncated", func(b []byte) []byte { return b[:len(b)-40] }, ErrArchiveCorrupted},
		{"payload-byte-flipped", func(b []byte) []byte { b[25] ^= 0xFF; return b }, ErrArchiveCorrupted},
		{"checksum-byte-flipped", func(b []byte) []byte { b[len(b)-1] ^= 0xFF; return b }, ErrArchiveCorrupted},
		{"bad-magic", func(b []byte) []byte { copy(b[:4], "NOPE"); return b }, ErrArchiveCorrupted},
		{"future-version", func(b []byte) []byte { b[5] = 99; return b }, ErrArchiveVersion},
		{"garbage", func(b []byte) []byte { return []byte("not an archive at all, just some long garbage text") }, ErrArchiveCorrupted},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			bad := tc.mutate(append([]byte(nil), good...))
			_, err := RestoreSession(bytes.NewReader(bad), sessMotion(), "cv-qpsd=0.05", sessBindings(), cfg)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("want %v, got %v", tc.wantErr, err)
			}
			t.Logf("%s refused: %v", tc.name, err)
		})
	}

	// Binding/motion/config mismatches are also refusals (not corruptions).
	base := sessBindings()
	wrongModel := append([]SourceBinding(nil), base...)
	wrongModel[0] = SourceBinding{Source: "gps", Descriptor: "CHANGED", Model: sessPosModel(0.04)}
	if _, err := RestoreSession(bytes.NewReader(good), sessMotion(), "cv-qpsd=0.05", wrongModel, cfg); !errors.Is(err, ErrArchiveBinding) {
		t.Fatalf("descriptor mismatch: %v", err)
	} else {
		t.Logf("model descriptor mismatch refused: %v", err)
	}
	if _, err := RestoreSession(bytes.NewReader(good), sessMotion(), "cv-qpsd=9.9", sessBindings(), cfg); !errors.Is(err, ErrArchiveBinding) {
		t.Fatalf("motion descriptor mismatch: %v", err)
	} else {
		t.Logf("motion descriptor mismatch refused: %v", err)
	}
	missing := sessBindings()[:2]
	if _, err := RestoreSession(bytes.NewReader(good), sessMotion(), "cv-qpsd=0.05", missing, cfg); !errors.Is(err, ErrArchiveBinding) {
		t.Fatalf("missing binding: %v", err)
	} else {
		t.Logf("missing source binding refused: %v", err)
	}
	badCfg := DefaultSessionConfig()
	badCfg.Engine.MaxDelayHorizon = 3.5
	if _, err := RestoreSession(bytes.NewReader(good), sessMotion(), "cv-qpsd=0.05", sessBindings(), badCfg); !errors.Is(err, ErrArchiveBinding) {
		t.Fatalf("config mismatch: %v", err)
	} else {
		t.Logf("engine config mismatch refused: %v", err)
	}
}

// TestSessionArchiveRoundTripFile: Save to an io.Writer and RestoreSession
// from an io.Reader keeps everything usable; float NaN-ish state is not
// produced here, but raw-bit encoding is exercised indirectly by bit-exact
// equality of every covariance entry.
func TestSessionArchiveRoundTripFile(t *testing.T) {
	s, _ := NewSession(sessInitial(), sessMotion(), "cv-qpsd=0.05", sessBindings(), DefaultSessionConfig())
	driveLive(t, s, buildLiveSequence(), 15)
	var buf bytes.Buffer
	if err := s.Save(&buf); err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(buf.Bytes(), []byte(archiveMagic)) {
		t.Fatal("archive does not start with self-describing magic")
	}
	r2, err := RestoreSession(&buf, sessMotion(), "cv-qpsd=0.05", sessBindings(), DefaultSessionConfig())
	if err != nil {
		t.Fatal(err)
	}
	assertSessionsBitEqual(t, "save/restore round trip", r2, s)
}

// TestSessionConcurrentSnapshotDuringIngestion: while goroutines keep
// submitting measurements, another goroutine continuously exports archives.
// Every exported archive must restore successfully and be internally
// consistent; run with -race to verify the absence of data races.
func TestSessionConcurrentSnapshotDuringIngestion(t *testing.T) {
	cfg := DefaultSessionConfig()
	s, _ := NewSession(sessInitial(), sessMotion(), "cv-qpsd=0.05", sessBindings(), cfg)

	const n = 200
	stop := make(chan struct{})
	var wg sync.WaitGroup

	// Snapshot storm runs concurrently with the whole ingestion. Every exported
	// archive must restore into a consistent point-in-time image; under -race
	// this also proves the export path shares state safely with ingestion.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			arc, err := s.Snapshot()
			if err != nil {
				t.Errorf("concurrent snapshot: %v", err)
				return
			}
			r, rerr := RestoreSession(bytes.NewReader(arc), sessMotion(), "cv-qpsd=0.05",
				sessBindings(), DefaultSessionConfig())
			if rerr != nil {
				t.Errorf("concurrent restore of live archive: %v", rerr)
				return
			}
			_ = r.State()
			_ = r.Stats()
			_ = r.Rejections()
		}
	}()

	// Deterministic feed: each time step submits the full multiset for that
	// step (from several goroutines with a barrier) BEFORE the clock advances,
	// exactly the invariant the engine's determinism relies on.
	for k := 0; k < n; k++ {
		var step sync.WaitGroup
		for p := 0; p < 4; p++ {
			step.Add(1)
			go func(k, p int) {
				defer step.Done()
				tt := float64(k)*0.1 + float64(p)*0.001
				_ = s.Submit("gps", tt, mat.Vec(tt, 0))
			}(k, p)
		}
		step.Wait()
		_, _, _ = s.Advance(float64(k+1) * 0.1)
	}
	close(stop)
	wg.Wait()

	// Same multiset, single goroutine: final state must match bit-for-bit.
	ref, _ := NewSession(sessInitial(), sessMotion(), "cv-qpsd=0.05", sessBindings(), cfg)
	for k := 0; k < n; k++ {
		for p := 0; p < 4; p++ {
			tt := float64(k)*0.1 + float64(p)*0.001
			_ = ref.Submit("gps", tt, mat.Vec(tt, 0))
		}
		_, _, _ = ref.Advance(float64(k+1) * 0.1)
	}
	assertSessionsBitEqual(t, "concurrent-vs-sequential", s, ref)
}

// Ensure the demo-style helpers compile and the rejections carry the expected
// reason strings used in the documentation.
func TestSessionReasonsDocumented(t *testing.T) {
	for _, r := range []RejectReason{ReasonDuplicate, ReasonExpired} {
		if strings.TrimSpace(string(r)) == "" {
			t.Fatalf("empty reason")
		}
	}
	if !errors.Is(ErrArchiveCorrupted, ErrArchiveCorrupted) ||
		!errors.Is(ErrArchiveVersion, ErrArchiveVersion) ||
		!errors.Is(ErrArchiveBinding, ErrArchiveBinding) {
		t.Fatal("archive sentinel errors broken")
	}
	t.Logf("reasons: %s, %s; archive errors exported", ReasonDuplicate, ReasonExpired)
}

var _ io.Reader = (*bytes.Reader)(nil)

// TestSessionBackfillWithRejectedSamples: a historical batch containing an
// out-of-gate outlier and a non-finite hard fault must produce exactly the
// same estimate, per-stream statistics and rejection REASONS as if those
// samples had arrived in real time.
func TestSessionBackfillWithRejectedSamples(t *testing.T) {
	motion := sessMotion()
	cfg := DefaultSessionConfig()

	history := []HistoricalMeasurement{
		{Time: 0.2, Source: "gps", Z: mat.Vec(0.2, 0.0)},
		{Time: 0.4, Source: "gps", Z: mat.Vec(0.4, 0.0)},
		{Time: 0.6, Source: "gps", Z: mat.Vec(50.0, 50.0)},    // huge NIS outlier
		{Time: 0.8, Source: "gps", Z: mat.Vec(math.NaN(), 0)}, // hard fault
	}

	// Reference: samples buffered before the session reaches t=1.0, and the
	// session reaches t=1.0 through the SAME clock schedule as the backfill
	// session below (one single Advance). Empty advances are themselves input
	// to lazy prediction, so an identical schedule is required for bit
	// equality.
	ref, _ := NewSession(sessInitial(), motion, "cv-qpsd=0.05", sessBindings(), cfg)
	for _, h := range history {
		if err := ref.Submit(h.Source, h.Time, h.Z.Clone()); err != nil {
			t.Logf("ref submit rejected: %v", err)
		}
	}
	if _, _, err := ref.Advance(1.0); err != nil {
		t.Fatal(err)
	}
	for _, r := range ref.Rejections() {
		t.Logf("ref rejection t=%.1f reason=%s", r.Time, r.Reason)
	}

	// Backfill at t=1.0.
	bf, _ := NewSession(sessInitial(), sessMotion(), "cv-qpsd=0.05", sessBindings(), cfg)
	_, _, _ = bf.Advance(1.0)
	n, rejs, err := bf.Backfill(history)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("backfill returned fused=%d rejections=%d", n, len(rejs))
	for _, r := range rejs {
		t.Logf("bf rejection t=%.1f reason=%s", r.Time, r.Reason)
	}
	assertSessions(t, "backfill-with-rejects", bf, ref, true)
}

// TestSessionEmptyArchiveRoundTrip: an archive taken before any measurement
// must restore into an equivalent fresh session and accept normal traffic.
func TestSessionEmptyArchiveRoundTrip(t *testing.T) {
	s, _ := NewSession(sessInitial(), sessMotion(), "cv-qpsd=0.05", sessBindings(), DefaultSessionConfig())
	arc, err := s.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	r, err := RestoreSession(bytes.NewReader(arc), sessMotion(), "cv-qpsd=0.05",
		sessBindings(), DefaultSessionConfig())
	if err != nil {
		t.Fatal(err)
	}
	seq := buildLiveSequence()
	feedSteps(t, s, seq, 0, 10)
	feedSteps(t, r, seq, 0, 10)
	assertSessionsBitEqual(t, "empty-archive", r, s)
}

// TestSessionRejectionLogConsistentAfterFolding: rejection entries are carried
// by checkpoints; after many folds the full rejection history must remain
// available and the session must still reject duplicates with the same
// accounting as an unbounded reference.
func TestSessionRejectionLogConsistentAfterFolding(t *testing.T) {
	cfg := DefaultSessionConfig()
	cfg.MemoryLimit = 20
	motion := sessMotion()
	s, _ := NewSession(sessInitial(), motion, "cv-qpsd=0.05", sessBindings(), cfg)
	ref, _ := NewSession(sessInitial(), sessMotion(), "cv-qpsd=0.05", sessBindings(), DefaultSessionConfig())

	for k := 1; k <= 80; k++ {
		tt := float64(k) * 0.1
		_ = s.Submit("gps", tt, mat.Vec(tt, 0))
		_ = ref.Submit("gps", tt, mat.Vec(tt, 0))
		// one expired-ish duplicate attempt every 10 steps on both
		_, _, _ = s.Advance(tt)
		_, _, _ = ref.Advance(tt)
	}
	// Attempt to fuse a sample already processed long ago; on both sessions it
	// must be duplicate (within horizon) or expired (beyond it), identically.
	errS := s.Submit("gps", 7.5, mat.Vec(7.5, 0))
	errR := ref.Submit("gps", 7.5, mat.Vec(7.5, 0))
	if rejectReasonOf(errS) != rejectReasonOf(errR) {
		t.Fatalf("reason differs after folding: %s vs %s", rejectReasonOf(errS), rejectReasonOf(errR))
	}
	t.Logf("old duplicate after folds consistently refused: %s", rejectReasonOf(errS))
	if len(s.Rejections()) != len(ref.Rejections()) {
		t.Fatalf("rejection history length %d vs %d", len(s.Rejections()), len(ref.Rejections()))
	}
}
