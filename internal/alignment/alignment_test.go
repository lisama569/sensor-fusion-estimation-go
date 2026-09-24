package alignment

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/models"
	"github.com/lisama569/sensor-fusion-estimation-go/internal/reject"
	"gonum.org/v1/gonum/mat"
)

func mk(sensor string, t float64, vals ...float64) models.Measurement {
	if len(vals) == 0 {
		vals = []float64{0}
	}
	return models.Measurement{SensorID: sensor, Timestamp: t, Z: mat.NewVecDense(len(vals), vals)}
}

func dumpSeq(t *testing.T, label string, ms []models.Measurement) {
	t.Helper()
	var b strings.Builder
	b.WriteString(fmt.Sprintf("input %s (%d):", label, len(ms)))
	for _, m := range ms {
		fmt.Fprintf(&b, " %s@%.1f", m.SensorID, m.Timestamp)
	}
	t.Log(b.String())
}

// Out-of-order arrival must be released in deterministic
// (timestamp, sensorID) order once the watermark covers it.
func TestOutOfOrderAlignment(t *testing.T) {
	in := []models.Measurement{
		mk("b", 2.0, 1),
		mk("a", 1.0, 1),
		mk("a", 2.0, 1),
		mk("b", 1.0, 1),
	}
	dumpSeq(t, "out-of-order", in)
	buf := NewBuffer(Config{MaxDelay: 0.5})
	for i, m := range in {
		if err := buf.Accept(m); err != nil {
			t.Fatalf("accept #%d %s@%.1f: %v", i, m.SensorID, m.Timestamp, err)
		}
	}
	out := buf.AdvanceTo(2.0)
	want := []struct {
		id string
		ts float64
	}{{"a", 1}, {"b", 1}, {"a", 2}, {"b", 2}}
	if len(out) != len(want) {
		t.Fatalf("released %d, want %d", len(out), len(want))
	}
	for i, w := range want {
		if out[i].SensorID != w.id || out[i].Timestamp != w.ts {
			t.Fatalf("position %d got %s@%.1f, want %s@%.1f",
				i, out[i].SensorID, out[i].Timestamp, w.id, w.ts)
		}
	}
	t.Logf("decision: released in (timestamp, sensorID) order: %s@%.1f %s@%.1f %s@%.1f %s@%.1f",
		out[0].SensorID, out[0].Timestamp, out[1].SensorID, out[1].Timestamp,
		out[2].SensorID, out[2].Timestamp, out[3].SensorID, out[3].Timestamp)
}

// A delayed measurement within MaxDelay is held then released; one beyond
// the window is rejected stale, not silently dropped.
func TestDelayedAndStale(t *testing.T) {
	buf := NewBuffer(Config{MaxDelay: 1.0})

	// Move watermark to t=5 first.
	if err := buf.Accept(mk("a", 5.0, 1)); err != nil {
		t.Fatal(err)
	}
	buf.AdvanceTo(5.0)
	t.Logf("decision: watermark advanced to 5.0; pending=%d", buf.Pending())

	// t=4.2 within window 5-1=4 -> accepted but held (older than watermark)
	if err := buf.Accept(mk("b", 4.2, 1)); err != nil {
		t.Fatalf("delayed-in-window should be accepted: %v", err)
	}
	t.Log("decision: b@4.2 accepted (delayed within MaxDelay=1.0)")

	// t=3.9 outside window -> stale
	err := buf.Accept(mk("b", 3.9, 1))
	if err == nil {
		t.Fatal("expected stale rejection")
	}
	if reject.Of(err) != reject.ReasonStaleTimestamp {
		t.Fatalf("want %s, got %v", reject.ReasonStaleTimestamp, err)
	}
	t.Logf("decision: b@3.9 rejected: %v", err)

	out := buf.AdvanceTo(6.0)
	if len(out) != 1 || out[0].SensorID != "b" || out[0].Timestamp != 4.2 {
		t.Fatalf("unexpected release: %+v", out)
	}
	t.Log("decision: advancing to 6.0 released held b@4.2")
}

// Exact (sensor, timestamp) duplicates are rejected even across watermarks.
func TestDuplicates(t *testing.T) {
	buf := NewBuffer(Config{MaxDelay: 0})
	m := mk("gps", 1.0, 7)
	if err := buf.Accept(m); err != nil {
		t.Fatal(err)
	}
	buf.AdvanceTo(1.0)
	if err := buf.Accept(m); err == nil {
		t.Fatal("duplicate must be rejected")
	} else if reject.Of(err) != reject.ReasonDuplicateTimestamp {
		t.Fatalf("want duplicate reason, got %v", err)
	} else {
		t.Logf("decision: re-submitting %s@%.1f rejected: %v", m.SensorID, m.Timestamp, err)
	}

	// Same timestamp on a different sensor is legal.
	if err := buf.Accept(mk("vel", 1.0, 7)); err != nil {
		t.Fatalf("other sensor at same timestamp must be accepted: %v", err)
	}
	t.Log("decision: vel@1.0 at same timestamp accepted (distinct channel)")
}

func TestRejectNonFiniteAndBadTimestamp(t *testing.T) {
	buf := NewBuffer(Config{MaxDelay: 1})
	cases := []struct {
		name   string
		m      models.Measurement
		reason reject.Reason
	}{
		{"nan value", mk("a", 1, math.NaN()), reject.ReasonNonFiniteValue},
		{"inf value", mk("a", 1, math.Inf(1)), reject.ReasonNonFiniteValue},
		{"nan time", models.Measurement{SensorID: "a", Timestamp: math.NaN(), Z: mat.NewVecDense(1, []float64{1})}, reject.ReasonBadTimestamp},
	}
	for _, tc := range cases {
		err := buf.Accept(tc.m)
		if err == nil || reject.Of(err) != tc.reason {
			t.Fatalf("%s: want reason %s, got %v", tc.name, tc.reason, err)
		}
		t.Logf("decision: %s rejected: %v", tc.name, err)
	}
}

// Same accept sequence after reset-like rebuild reproduces identical order.
func TestReproducibleOrder(t *testing.T) {
	build := func() []models.Measurement {
		buf := NewBuffer(Config{MaxDelay: 0.5})
		in := []models.Measurement{mk("c", 1), mk("a", 2), mk("b", 1), mk("a", 1), mk("c", 2), mk("b", 2)}
		for _, m := range in {
			if err := buf.Accept(m); err != nil {
				t.Fatal(err)
			}
		}
		return buf.AdvanceTo(2)
	}
	r1, r2 := build(), build()
	for i := range r1 {
		if r1[i].SensorID != r2[i].SensorID || r1[i].Timestamp != r2[i].Timestamp {
			t.Fatalf("non-reproducible order at %d", i)
		}
	}
	t.Log("decision: two runs with identical inputs produce identical release order")
}
