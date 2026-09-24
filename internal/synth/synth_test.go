package synth

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

func specs() []ChannelSpec {
	return []ChannelSpec{
		{ID: "gps", Kind: "gps", Period: 1, Delay: 0.1, Noise2D: [2][2]float64{{0.5, 0}, {0, 0.5}}},
		{ID: "vel", Kind: "velocity", Period: 0.5, Delay: 0, Noise2D: [2][2]float64{{0.1, 0}, {0, 0.1}}},
		{ID: "polar", Kind: "polar", Period: 0.8, Delay: 0.2, Noise2D: [2][2]float64{{0.2, 0}, {0, 0.01}},
			Beacon: [2]float64{3, 4}},
	}
}

func TestDeterministicSameSeed(t *testing.T) {
	mk := func() []float64 {
		g := NewGenerator(123, 0.1, 4, []float64{0, 1, 0, 0}, specs())
		ms := g.Sample()
		var flat []float64
		for _, m := range ms {
			flat = append(flat, m.Timestamp)
			for i := 0; i < m.Z.Len(); i++ {
				flat = append(flat, m.Z.AtVec(i))
			}
		}
		return flat
	}
	a, b := mk(), mk()
	if len(a) != len(b) {
		t.Fatalf("length mismatch %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-reproducible value at %d: %v vs %v", i, a[i], b[i])
		}
	}
	t.Logf("decision: same seed reproduces %d samples exactly", len(a))
}

func TestDifferentSeedsDiffer(t *testing.T) {
	ga := NewGenerator(1, 0.1, 10, []float64{0, 1, 0, 0}, specs())
	gb := NewGenerator(2, 0.1, 10, []float64{0, 1, 0, 0}, specs())
	za, zb := ga.Sample(), gb.Sample()
	diff := 0
	for i := range za {
		for k := 0; k < za[i].Z.Len(); k++ {
			if za[i].Z.AtVec(k) != zb[i].Z.AtVec(k) {
				diff++
			}
		}
	}
	if diff == 0 {
		t.Fatal("different seeds produced identical noise")
	}
	t.Logf("decision: different seeds differ in %d noisy components", diff)
}

func TestTimestampAndDelaySemantics(t *testing.T) {
	g := NewGenerator(1, 0.1, 2, []float64{0, 1, 0, 0}, specs())
	ms := g.Sample()
	var b strings.Builder
	for _, m := range ms {
		fmt.Fprintf(&b, " %s@%.2f(arr~%.2f)", m.SensorID, m.Timestamp, g.Arrival(m))
	}
	t.Logf("input sequence:%s", b.String())
	for _, m := range ms {
		if g.Arrival(m) < m.Timestamp {
			t.Fatalf("%s arrival %.3f must be >= timestamp %.3f", m.SensorID, g.Arrival(m), m.Timestamp)
		}
		// timestamps must be exact multiples of the channel period
		var period float64
		for _, sp := range specs() {
			if sp.ID == m.SensorID {
				period = sp.Period
			}
		}
		r := math.Mod(math.Abs(m.Timestamp), period)
		if r > 1e-9 && math.Abs(r-period) > 1e-9 {
			t.Fatalf("%s timestamp %.3f not on period %.2f grid", m.SensorID, m.Timestamp, period)
		}
	}
	t.Log("decision: timestamps are on each channel's period grid; arrivals = ts + delay")
}

func TestCorruptWindow(t *testing.T) {
	g := NewGenerator(5, 0.1, 10, []float64{0, 1, 0, 0}, specs())
	ms := g.Sample()
	ms = g.Corrupt(ms, "gps", 2, 5, true)
	nan, clean := 0, 0
	for _, m := range ms {
		if m.SensorID != "gps" {
			continue
		}
		isNaN := math.IsNaN(m.Z.AtVec(0))
		if m.Timestamp >= 2 && m.Timestamp < 5 {
			if !isNaN {
				t.Fatalf("gps@%.2f inside window should be NaN", m.Timestamp)
			}
			nan++
		} else if isNaN {
			t.Fatalf("gps@%.2f outside window should be clean", m.Timestamp)
		} else {
			clean++
		}
	}
	if nan == 0 {
		t.Fatal("no samples corrupted")
	}
	t.Logf("decision: corrupted %d gps samples in [2,5), %d clean elsewhere", nan, clean)
}
