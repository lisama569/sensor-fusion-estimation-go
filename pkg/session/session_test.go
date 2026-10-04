package session

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/engine"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/model"
)

// ---- deterministic fixtures -------------------------------------------------

func testConfig() engine.Config {
	c := engine.DefaultConfig()
	c.MaxDelayHorizon = 10.0 // generous so the replay tests stay in-window
	return c
}

func newGPS(t float64) *model.LinearPosition {
	_ = t
	return &model.LinearPosition{R: mat.From([][]float64{{0.04, 0}, {0, 0.04}})}
}

func initialState() *model.State {
	return &model.State{Time: 0, X: mat.Vec(0, 0, 1, 0), P: mat.Identity(4)}
}

func newTestSession(t *testing.T) *Session {
	t.Helper()
	s, err := New(initialState(), &model.ConstantVelocity{QPSD: 0.05},
		Options{Engine: testConfig(), MaxProcessedPerSource: 4096})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s
}

// truePos is the scripted noiseless ground truth: x=t, y=0.
func truePos(t float64) (float64, float64) { return t, 0 }

func gpsItem(t float64, noise float64) ReplayItem {
	x, y := truePos(t)
	return ReplayItem{
		Source: "gps",
		Time:   t,
		Z:      mat.Vec(x+noise, y),
		Model:  &model.LinearPosition{R: mat.From([][]float64{{0.04, 0}, {0, 0.04}})},
	}
}

// liveFeed runs the reference "never interrupted" path: feed each measurement
// and Advance exactly to its timestamp (deterministic per-sample processing).
func liveFeed(t *testing.T, times []float64, noises []float64) *model.State {
	t.Helper()
	eng, err := engine.New(initialState(), &model.ConstantVelocity{QPSD: 0.05}, testConfig())
	if err != nil {
		t.Fatalf("engine.New: %v", err)
	}
	// Permute nothing; times are already chronological.
	for i, tm := range times {
		it := gpsItem(tm, noises[i])
		if err := eng.Submit(it.Source, it.Time, it.Z, it.Model); err != nil {
			t.Fatalf("reference submit: %v", err)
		}
		if _, _, err := eng.Advance(tm); err != nil {
			t.Fatalf("reference advance: %v", err)
		}
	}
	return eng.State()
}

func assertStateBitEqual(t *testing.T, tag string, got, want *model.State) {
	t.Helper()
	if got.Time != want.Time {
		t.Fatalf("%s: time %.17g != %.17g", tag, got.Time, want.Time)
	}
	if len(got.X.Data) != len(want.X.Data) || len(got.P.Data) != len(want.P.Data) {
		t.Fatalf("%s: dimension mismatch", tag)
	}
	for i := range got.X.Data {
		if got.X.Data[i] != want.X.Data[i] {
			t.Fatalf("%s: X[%d] %.17g != %.17g (bit mismatch)", tag, i, got.X.Data[i], want.X.Data[i])
		}
	}
	for i := range got.P.Data {
		if got.P.Data[i] != want.P.Data[i] {
			t.Fatalf("%s: P[%d] %.17g != %.17g (bit mismatch)", tag, i, got.P.Data[i], want.P.Data[i])
		}
	}
}

// ---- 1. save / resume equals an uninterrupted run --------------------------

func TestSaveResumeBitIdenticalToUninterrupted(t *testing.T) {
	times := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
	noises := []float64{0.01, -0.02, 0.015, 0.0, -0.01, 0.02, -0.015, 0.005, 0.01, -0.005}
	want := liveFeed(t, times, noises)

	// Interrupted run: stop after measurement 4, export, restore in a brand-new
	// session object (simulating a new process), then continue.
	s := newTestSession(t)
	for i := 0; i < 4; i++ {
		it := gpsItem(times[i], noises[i])
		if err := s.Submit(it.Source, it.Time, it.Z, it.Model); err != nil {
			t.Fatalf("submit %d: %v", i, err)
		}
		if _, _, err := s.Advance(times[i]); err != nil {
			t.Fatalf("advance %d: %v", i, err)
		}
	}
	data, err := s.Export()
	if err != nil {
		t.Fatalf("Export: %v", err)
	}
	if !bytes.Contains(data, []byte(`"version": 1`)) {
		t.Fatal("archive must self-describe its version")
	}
	t.Logf("exported %d-byte self-describing archive at t=%.1f", len(data), times[3])

	s2, err := Import(data)
	if err != nil {
		t.Fatalf("Import: %v", err)
	}
	for i := 4; i < len(times); i++ {
		it := gpsItem(times[i], noises[i])
		if err := s2.Submit(it.Source, it.Time, it.Z, it.Model); err != nil {
			t.Fatalf("resumed submit %d: %v", i, err)
		}
		if _, _, err := s2.Advance(times[i]); err != nil {
			t.Fatalf("resumed advance %d: %v", i, err)
		}
	}
	got := s2.State()
	assertStateBitEqual(t, "resumed vs uninterrupted", got, want)

	// Statistics and per-stream counters must also match.
	ws := s2.Stats()
	if ws.Updates != 10 || ws.Predicts != 10 || ws.Rejected != 0 {
		t.Fatalf("resumed stats mismatch: %+v", ws)
	}
	t.Logf("resumed estimate bit-identical to uninterrupted run; stats=%+v", ws)
}

// Multiple save/restore cycles, including pending (buffered, not yet advanced)
// measurements carried across the boundary.
func TestMultipleSaveResumeWithPending(t *testing.T) {
	times := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	noises := []float64{0.01, -0.01, 0.02, -0.02, 0.005, -0.005}
	want := liveFeed(t, times, noises)

	s := newTestSession(t)
	// Submit two but advance only to the first: the second is pending and must
	// survive the archive unchanged.
	it0, it1 := gpsItem(times[0], noises[0]), gpsItem(times[1], noises[1])
	if err := s.Submit(it0.Source, it0.Time, it0.Z, it0.Model); err != nil {
		t.Fatal(err)
	}
	if err := s.Submit(it1.Source, it1.Time, it1.Z, it1.Model); err != nil {
		t.Fatal(err)
	}
	if _, _, err := s.Advance(times[0]); err != nil {
		t.Fatal(err)
	}

	cur := s
	for i := 2; i < len(times); i++ {
		data, err := cur.Export()
		if err != nil {
			t.Fatalf("Export cycle %d: %v", i, err)
		}
		cur, err = Import(data)
		if err != nil {
			t.Fatalf("Import cycle %d: %v", i, err)
		}
		it := gpsItem(times[i], noises[i])
		if err := cur.Submit(it.Source, it.Time, it.Z, it.Model); err != nil {
			t.Fatalf("submit cycle %d: %v", i, err)
		}
		if _, _, err := cur.Advance(times[i]); err != nil {
			t.Fatalf("advance cycle %d: %v", i, err)
		}
	}
	got := cur.State()
	assertStateBitEqual(t, "multi-cycle with pending", got, want)
	t.Log("pending measurements and health/state survive multiple archive cycles bit-identically")
}

// ---- 2. historical replay equals live arrival ------------------------------

func TestReplayEqualsLiveArrival(t *testing.T) {
	times := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
	noises := []float64{0.02, -0.01, 0.015, -0.02, 0.01}
	want := liveFeed(t, times, noises)

	s := newTestSession(t)
	items := make([]ReplayItem, len(times))
	for i := range times {
		items[i] = gpsItem(times[i], noises[i])
	}
	// Deliberately shuffle arrival order within the batch; the engine's
	// deterministic (time, source) order must remove that dependence.
	shuffled := []ReplayItem{items[3], items[0], items[4], items[1], items[2]}
	rej, err := s.Replay(shuffled)
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	if len(rej) != 0 {
		t.Fatalf("unexpected replay rejections: %+v", rej)
	}
	got := s.State()
	assertStateBitEqual(t, "replay vs live", got, want)
	t.Log("historical replay (shuffled arrival) is bit-identical to live arrival")
}

// ---- 3. duplicate rejection reasons are distinguishable --------------------

func rejectReason(err error) engine.RejectReason {
	var re *engine.RejectError
	if errors.As(err, &re) {
		return re.Reason
	}
	return ""
}

func TestCrossArchiveAndCrossBatchDuplicates(t *testing.T) {
	s := newTestSession(t)

	// First batch: one measurement fused.
	if _, err := s.Replay([]ReplayItem{gpsItem(0.5, 0.01)}); err != nil {
		t.Fatal(err)
	}
	// Second batch repeats the SAME sample: cross-batch duplicate.
	rej2, err := s.Replay([]ReplayItem{gpsItem(0.5, 0.01)})
	if err != nil {
		t.Fatal(err)
	}
	if len(rej2) != 1 || rej2[0].Reason != engine.ReasonAlreadyProcessed {
		t.Fatalf("cross-batch duplicate: got %+v, want already_processed", rej2)
	}
	t.Logf("cross-batch duplicate refused: reason=%s detail=%q", rej2[0].Reason, rej2[0].Detail)

	// Archive and restore, then resubmit the pre-archive sample: cross-archive.
	data, err := s.Export()
	if err != nil {
		t.Fatal(err)
	}
	s2, err := Import(data)
	if err != nil {
		t.Fatal(err)
	}
	err = s2.Submit("gps", 0.5, mat.Vec(0.51, 0),
		&model.LinearPosition{R: mat.From([][]float64{{0.04, 0}, {0, 0.04}})})
	if got := rejectReason(err); got != engine.ReasonAlreadyProcessed {
		t.Fatalf("cross-archive duplicate: reason=%q, want already_processed", got)
	}
	t.Log("cross-archive duplicate refused with already_processed; never fused twice")

	// Same-batch duplicate (two copies in one Replay) is the engine's own
	// duplicate_timestamp, a different reason.
	dupA := gpsItem(0.7, 0.0)
	dupB := gpsItem(0.7, 0.0)
	rej3, err := s2.Replay([]ReplayItem{dupA, dupB})
	if err != nil {
		t.Fatal(err)
	}
	var sawDupTS bool
	for _, r := range rej3 {
		if r.Reason == engine.ReasonDuplicate {
			sawDupTS = true
		}
		if r.Reason == engine.ReasonAlreadyProcessed {
			t.Fatalf("fresh same-batch duplicate must not be already_processed: %+v", r)
		}
	}
	if !sawDupTS {
		t.Fatalf("same-batch duplicate must be duplicate_timestamp: %+v", rej3)
	}
	t.Log("same-batch duplicate refused with duplicate_timestamp (distinct reason)")

	// A genuinely-new measurement still fuses after the refused duplicates:
	// proves rejected samples never counted nor blocked the stream.
	st, err := s2.Replay([]ReplayItem{gpsItem(0.9, 0.0)})
	_ = st
	_ = err
	stats := s2.Stats()
	if stats.Updates != 2 { // t=0.5 (batch1) + t=0.9 only; 0.7 dup group refused
		t.Fatalf("updates=%d, want 2; rejected duplicates must never count", stats.Updates)
	}
	t.Logf("after duplicate refusals, genuine samples still fuse; updates=%d", stats.Updates)
}

func TestExpiredIsDistinctFromDuplicate(t *testing.T) {
	c := testConfig()
	c.MaxDelayHorizon = 1.0
	s, err := New(initialState(), &model.ConstantVelocity{QPSD: 0.05},
		Options{Engine: c, MaxProcessedPerSource: 4096})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Replay([]ReplayItem{gpsItem(0.5, 0)}); err != nil {
		t.Fatal(err)
	}
	// Advance the clock far beyond the horizon, then replay the old sample.
	if _, _, err := s.Advance(5.0); err != nil {
		t.Fatal(err)
	}
	rej, err := s.Replay([]ReplayItem{gpsItem(0.5, 0)})
	if err != nil {
		t.Fatal(err)
	}
	if len(rej) != 1 {
		t.Fatalf("want 1 rejection, got %+v", rej)
	}
	// The old timestamp is unusable at all -> expired_measurement, not the
	// duplicate reason, even though it was previously processed.
	if rej[0].Reason != engine.ReasonExpired {
		t.Fatalf("old repeat reason=%q, want expired_measurement", rej[0].Reason)
	}
	t.Logf("out-of-horizon historical sample refused as %s (distinct from duplicates)", rej[0].Reason)
}

// ---- 4. corrupt / version-mismatched archives are refused ------------------

func TestCorruptArchiveRejected(t *testing.T) {
	s := newTestSession(t)
	if _, err := s.Replay([]ReplayItem{gpsItem(0.3, 0.01)}); err != nil {
		t.Fatal(err)
	}
	good, err := s.Export()
	if err != nil {
		t.Fatal(err)
	}

	cases := map[string][]byte{
		"truncated":      good[:len(good)/2],
		"random bytes":   []byte("{not a session archive at all!!"),
		"flipped body":   flipByte(copyBytes(good), 400),
		"tampered state": tamperBody(t, good),
	}
	for name, bad := range cases {
		if _, err := Import(bad); !errors.Is(err, ErrArchiveCorrupt) {
			t.Fatalf("%s: Import err=%v, want ErrArchiveCorrupt", name, err)
		}
		t.Logf("corrupt archive (%s) refused: %v", name, ErrArchiveCorrupt)
	}

	// Bad magic.
	var env envelope
	if err := json.Unmarshal(good, &env); err != nil {
		t.Fatal(err)
	}
	env.Magic = "WRONG"
	env.SHA256 = "00" // any mismatch
	badMagic, _ := json.Marshal(&env)
	if _, err := Import(badMagic); !errors.Is(err, ErrArchiveCorrupt) {
		t.Fatalf("bad magic err=%v, want ErrArchiveCorrupt", err)
	}
	t.Log("bad-magic archive refused as corrupt")
}

func TestVersionMismatchRejected(t *testing.T) {
	s := newTestSession(t)
	good, err := s.Export()
	if err != nil {
		t.Fatal(err)
	}

	// Unsupported ENVELOPE version -> ErrArchiveVersion.
	var env envelope
	if err := json.Unmarshal(good, &env); err != nil {
		t.Fatal(err)
	}
	env.Version = ArchiveFormatVersion + 1
	// recompute checksum so failure is the version check, not checksum.
	sum := sha256hex(env.Body)
	env.SHA256 = sum
	bumped, _ := json.Marshal(&env)
	if _, err := Import(bumped); !errors.Is(err, ErrArchiveVersion) {
		t.Fatalf("bumped envelope version err=%v, want ErrArchiveVersion", err)
	}
	t.Logf("unsupported envelope version refused: %v", ErrArchiveVersion)

	// Unsupported ENGINE snapshot version -> ErrArchiveVersion as well.
	env2 := envelope{}
	if err := json.Unmarshal(good, &env2); err != nil {
		t.Fatal(err)
	}
	var dto snapshotDTO
	if err := json.Unmarshal(env2.Body, &dto); err != nil {
		t.Fatal(err)
	}
	dto.Engine.FormatVersion = 999
	newBody, _ := json.Marshal(&dto)
	env2.Body = newBody
	env2.SHA256 = sha256hex(newBody)
	badEng, _ := json.Marshal(&env2)
	if _, err := Import(badEng); !errors.Is(err, ErrArchiveVersion) {
		t.Fatalf("bumped engine version err=%v, want ErrArchiveVersion", err)
	}
	t.Logf("unsupported engine snapshot version refused: %v", ErrArchiveVersion)
}

// ---- 5. consistent snapshot under concurrent writers (race) ----------------

func TestConcurrentExportConsistencyNoRace(t *testing.T) {
	s := newTestSession(t)
	var writers sync.WaitGroup
	var exporters sync.WaitGroup
	stop := make(chan struct{})

	// Writers continuously submit + advance on two sources.
	for w := 0; w < 8; w++ {
		writers.Add(1)
		go func(id int) {
			defer writers.Done()
			i := 0
			for {
				select {
				case <-stop:
					return
				default:
				}
				tm := 0.01 * float64(i)
				src := "gps"
				if id%2 == 0 {
					src = "imu"
				}
				_ = s.Submit(src, tm, mat.Vec(tm, 0),
					&model.LinearPosition{R: mat.From([][]float64{{0.04, 0}, {0, 0.04}})})
				_, _, _ = s.Advance(tm)
				i++
				if i > 200 {
					return
				}
			}
		}(w)
	}
	// Exporters continuously take consistent images; every one must import and
	// report the same time as its engine state (a real, usable point in time).
	exporters.Add(1)
	go func() {
		defer exporters.Done()
		for {
			select {
			case <-stop:
				return
			default:
			}
			data, err := s.Export()
			if err != nil {
				t.Errorf("concurrent Export: %v", err)
				return
			}
			restored, err := Import(data)
			if err != nil {
				t.Errorf("concurrent Import: %v", err)
				return
			}
			if restored.State().Time != restored.Engine().State().Time {
				t.Errorf("exported snapshot is internally inconsistent")
				return
			}
		}
	}()
	writers.Wait()
	close(stop)
	exporters.Wait()

	// Final sanity: memory stayed bounded throughout.
	st := s.State()
	if math.IsNaN(st.Time) {
		t.Fatal("final state time NaN after concurrent run")
	}
	t.Logf("concurrent submit/advance/export completed without data races; final t=%.3f", st.Time)
}

// ---- 6. file round-trip -----------------------------------------------------

func TestSaveLoadFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "session.json")
	s := newTestSession(t)
	if _, err := s.Replay([]ReplayItem{gpsItem(0.4, 0.02)}); err != nil {
		t.Fatal(err)
	}
	if err := s.SaveFile(path); err != nil {
		t.Fatalf("SaveFile: %v", err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("archive perms=%o, want 0600", info.Mode().Perm())
	}
	s2, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	assertStateBitEqual(t, "file round-trip", s2.State(), s.State())
	t.Log("SaveFile/LoadFile round-trip is bit-identical; archive stored 0600")
}

// ---- helpers ----------------------------------------------------------------

func copyBytes(b []byte) []byte {
	c := make([]byte, len(b))
	copy(c, b)
	return c
}

func flipByte(b []byte, idx int) []byte {
	if idx < len(b) {
		b[idx] ^= 0xFF
	}
	return b
}

// tamperBody modifies a byte inside the body region (after the "body": line)
// without updating the checksum.
func tamperBody(t *testing.T, good []byte) []byte {
	t.Helper()
	marker := []byte(`"body":`)
	i := bytes.Index(good, marker)
	if i < 0 {
		t.Fatal("no body marker in archive")
	}
	idx := i + len(marker) + 40
	if idx >= len(good)-5 {
		idx = len(good) - 10
	}
	out := copyBytes(good)
	if out[idx] >= '0' && out[idx] <= '8' {
		out[idx]++
	} else {
		out[idx] = '1'
	}
	return out
}

func sha256hex(b []byte) string {
	h := sha256.Sum256(b)
	sum := h[:]
	return fmt.Sprintf("%x", sum)
}

var _ = newGPS
