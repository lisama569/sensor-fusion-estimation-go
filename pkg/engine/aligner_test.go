package engine

import (
	"fmt"
	"math"
	"testing"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/model"
)

func mkMeas(t float64, src string) *Measurement {
	return &Measurement{
		Time:   t,
		Source: src,
		Z:      mat.Vec(0.0, 0.0),
		Model:  &model.LinearPosition{R: diag2(1, 1)},
	}
}

func num(v float64) string { return fmt.Sprintf("%.1f", v) }

func TestAlignerOutOfOrderThenOrdered(t *testing.T) {
	a := newAligner(0, 2.0)
	in := []*Measurement{mkMeas(0.8, "b"), mkMeas(0.4, "a"), mkMeas(1.2, "a"), mkMeas(0.4, "b")}
	for _, m := range in {
		if err := a.submit(m); err != nil {
			t.Fatalf("submit %s@%.1f: %v", m.Source, m.Time, err)
		}
	}
	due, rejected := a.advance(1.0)
	if len(rejected) != 0 {
		t.Fatalf("unexpected rejects: %v", rejected)
	}
	got := make([][2]string, 0)
	for _, m := range due {
		got = append(got, [2]string{num(m.Time), m.Source})
	}
	want := [][2]string{{"0.4", "a"}, {"0.4", "b"}, {"0.8", "b"}}
	if len(got) != len(want) {
		t.Fatalf("due len=%d want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("order mismatch: got %v want %v", got, want)
		}
	}
}

func TestAlignerDuplicateAndExpiredDecidedAtAdvance(t *testing.T) {
	a := newAligner(0, 1.0)
	// Two identical (source,time) submitted in arbitrary order.
	a.submit(mkMeas(0.5, "a"))
	a.submit(mkMeas(0.5, "a"))
	a.submit(mkMeas(0.5, "b")) // other source, same time, fine
	due, rejected := a.advance(1.0)
	if len(due) != 1 || due[0].Source != "b" {
		t.Fatalf("want only b due, got %v", due)
	}
	if len(rejected) != 2 {
		t.Fatalf("duplicate group: both copies must be refused, got %v", rejected)
	}
	for _, r := range rejected {
		if r.Reason != ReasonDuplicate {
			t.Fatalf("want duplicate reason, got %v", r)
		}
	}

	// After committing t=3 with last committed a-time 1.2, a t=0.5 resubmit
	// is expired (0.5 < 1.2-1.0=0.2 false... use a clearly older one), while
	// a t=0.9 sample is still inside the horizon (0.9 >= 1.2-1.0=0.2) and is a
	// legal late measurement.
	// Global committed clock becomes 3; horizon 1 -> expiry bound 2.
	a.submit(mkMeas(1.0, "a")) // 1.0 < 2 -> expired
	a.submit(mkMeas(2.5, "a")) // >= 2 -> late but still in horizon, valid
	due2, rej2 := a.advance(3.0)
	var gotExp, gotLate bool
	for _, r := range rej2 {
		if r.Reason == ReasonExpired {
			gotExp = true
		}
	}
	for _, m := range due2 {
		if m.Source == "a" && math.Abs(m.Time-2.5) < 1e-9 {
			gotLate = true
		}
	}
	if !gotExp {
		t.Fatalf("expected expired rejection, got %v", rej2)
	}
	if !gotLate {
		t.Fatal("0.9 should be accepted as in-horizon late measurement")
	}

	// Two identical timestamps in the same later batch -> one duplicate.
	a.submit(mkMeas(3.5, "a"))
	a.submit(mkMeas(3.5, "a"))
	_, rej3 := a.advance(4.0)
	dup := 0
	for _, r := range rej3 {
		if r.Reason == ReasonDuplicate {
			dup++
		}
	}
	if dup != 2 {
		t.Fatalf("duplicate group: both copies refused, want 2 got %v", rej3)
	}
}

func TestAlignerBadTimestampImmediate(t *testing.T) {
	a := newAligner(0, 1.0)
	err := a.submit(mkMeas(math.NaN(), "a"))
	re, ok := err.(*RejectError)
	if !ok || re.Reason != ReasonBadTimestamp {
		t.Fatalf("want bad_timestamp, got %v", err)
	}
}
