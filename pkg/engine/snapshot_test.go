package engine

import (
	"math"
	"testing"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/model"
)

func gpsModel() *model.LinearPosition {
	return &model.LinearPosition{R: mat.From([][]float64{{0.04, 0}, {0, 0.04}})}
}

// TestSnapshotRoundTripBitIdentical: a snapshot taken mid-flight (including a
// buffered pending measurement and a faulted stream) restores to an engine
// whose continued run is bit-identical to the engine that never restarted.
func TestSnapshotRoundTripBitIdentical(t *testing.T) {
	cfg := DefaultConfig()
	mk := func() (*Engine, error) {
		return New(&model.State{Time: 0, X: mat.Vec(0, 0, 1, 0), P: mat.Identity(4)},
			&model.ConstantVelocity{QPSD: 0.05}, cfg)
	}

	ref, err := mk()
	if err != nil {
		t.Fatal(err)
	}
	// Reference uninterrupted sequence.
	for i := 1; i <= 6; i++ {
		tm := 0.1 * float64(i)
		if err := ref.Submit("gps", tm, mat.Vec(tm, 0), gpsModel()); err != nil {
			t.Fatal(err)
		}
		if _, _, err := ref.Advance(tm); err != nil {
			t.Fatal(err)
		}
	}

	// Interrupted engine, snapshotted after 3 advances with one pending sample.
	a, err := mk()
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 3; i++ {
		tm := 0.1 * float64(i)
		if err := a.Submit("gps", tm, mat.Vec(tm, 0), gpsModel()); err != nil {
			t.Fatal(err)
		}
		if _, _, err := a.Advance(tm); err != nil {
			t.Fatal(err)
		}
	}
	// Submit #4 but do not advance: it must be carried as pending.
	if err := a.Submit("gps", 0.4, mat.Vec(0.4, 0), gpsModel()); err != nil {
		t.Fatal(err)
	}
	snap, err := a.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	if len(snap.Pending) != 1 || snap.Pending[0].Time != 0.4 {
		t.Fatalf("pending snapshot wrong: %+v", snap.Pending)
	}
	b, err := NewFromSnapshot(snap)
	if err != nil {
		t.Fatalf("NewFromSnapshot: %v", err)
	}
	// Continue the same sequence; the pending #4 fuses at Advance(0.4).
	for i := 4; i <= 6; i++ {
		tm := 0.1 * float64(i)
		if i > 4 {
			if err := b.Submit("gps", tm, mat.Vec(tm, 0), gpsModel()); err != nil {
				t.Fatal(err)
			}
		}
		if _, _, err := b.Advance(tm); err != nil {
			t.Fatal(err)
		}
	}

	rs, bs := ref.State(), b.State()
	if rs.Time != bs.Time {
		t.Fatalf("time %.17g != %.17g", rs.Time, bs.Time)
	}
	for i := range rs.X.Data {
		if rs.X.Data[i] != bs.X.Data[i] {
			t.Fatalf("X[%d] %.17g != %.17g", i, rs.X.Data[i], bs.X.Data[i])
		}
	}
	for i := range rs.P.Data {
		if rs.P.Data[i] != bs.P.Data[i] {
			t.Fatalf("P[%d] %.17g != %.17g", i, rs.P.Data[i], bs.P.Data[i])
		}
	}
	rr, br := ref.Stats(), b.Stats()
	if rr.Updates != br.Updates || rr.Predicts != br.Predicts || rr.Rejected != br.Rejected {
		t.Fatalf("stats mismatch ref=%+v restored=%+v", rr, br)
	}
	t.Logf("engine snapshot round-trip bit-identical; updates=%d predicts=%d rejected=%d",
		br.Updates, br.Predicts, br.Rejected)
}

// TestNewFromSnapshotRejectsBadVersion ensures an engine snapshot from another
// layout version is refused rather than partially loaded.
func TestNewFromSnapshotRejectsBadVersion(t *testing.T) {
	e, err := New(&model.State{Time: 0, X: mat.Vec(0, 0, 0, 0), P: mat.Identity(4)},
		&model.ConstantVelocity{QPSD: 1}, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	snap, err := e.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	snap.FormatVersion++
	if _, err := NewFromSnapshot(snap); err == nil {
		t.Fatal("expected version mismatch error, got nil")
	}
	t.Log("engine refuses a snapshot with an incompatible layout version")
}

// TestSnapshotCarriesFaultedStream: a hard-faulted isolated stream and its
// counters survive a snapshot round-trip.
func TestSnapshotCarriesFaultedStream(t *testing.T) {
	cfg := DefaultConfig()
	mk := func() *Engine {
		e, err := New(&model.State{Time: 0, X: mat.Vec(0, 0, 0, 0), P: mat.Identity(4)},
			&model.ConstantVelocity{QPSD: 1}, cfg)
		if err != nil {
			t.Fatal(err)
		}
		return e
	}
	e := mk()
	nan := math.NaN()
	if err := e.Submit("radar", 0.1, mat.Vec(nan, 0), &model.RangeBearing{
		Sx: 5, Sy: 5, R: mat.From([][]float64{{0.01, 0}, {0, 0.01}})}); err != nil {
		t.Fatal(err)
	}
	if _, rej, err := e.Advance(0.1); err != nil {
		t.Fatal(err)
	} else if len(rej) != 1 || rej[0].Reason != ReasonNonFinite {
		t.Fatalf("want non_finite_value rejection, got %+v", rej)
	}
	if !e.ht.faulty("radar") {
		t.Fatal("radar stream should be isolated after NaN sample")
	}

	snap, err := e.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	h, ok := snap.Health["radar"]
	if !ok || !h.Faulty {
		t.Fatalf("snapshot must carry faulty radar health, got %+v ok=%v", h, ok)
	}
	r, err := NewFromSnapshot(snap)
	if err != nil {
		t.Fatal(err)
	}
	if !r.ht.faulty("radar") {
		t.Fatal("restored engine must still isolate radar")
	}
	if got := r.Stats().PerStream["radar"].HardFaults; got != 1 {
		t.Fatalf("restored hard faults=%d, want 1", got)
	}
	t.Log("fault state machine and per-stream hard-fault counters survive snapshot restore")
}
