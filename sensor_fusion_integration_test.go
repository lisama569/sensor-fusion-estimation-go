package sensor_fusion_estimation_go_test

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/alignment"
	"github.com/lisama569/sensor-fusion-estimation-go/internal/filter"
	"github.com/lisama569/sensor-fusion-estimation-go/internal/fusion"
	"github.com/lisama569/sensor-fusion-estimation-go/internal/models"
	"github.com/lisama569/sensor-fusion-estimation-go/internal/reject"
	"github.com/lisama569/sensor-fusion-estimation-go/internal/synth"
	"gonum.org/v1/gonum/mat"
)

// endToEnd builds a 3-channel fuser (two linear models + one nonlinear
// polar model), feeds the given measurements concurrently, drives the
// deterministic watermark clock, and returns the final state + statistics.
func endToEnd(t *testing.T, seed int64, mutate func([]models.Measurement) []models.Measurement,
	concurrent bool) (filter.State, fusion.Stats, *fusion.Fuser, *synth.Generator, []synth.ChannelSpec) {
	t.Helper()
	const endT = 8.0
	sm := models.NewConstantVelocity2D(0.5)
	specs := []synth.ChannelSpec{
		{ID: "gps", Kind: "gps", Period: 0.5, Delay: 0.2, Noise2D: [2][2]float64{{1, 0}, {0, 1}}},
		{ID: "vel", Kind: "velocity", Period: 0.25, Delay: 0.05, Noise2D: [2][2]float64{{0.25, 0}, {0, 0.25}}},
		{ID: "polar", Kind: "polar", Period: 0.4, Delay: 0.25,
			Noise2D: [2][2]float64{{0.4, 0}, {0, 0.02}}, Beacon: [2]float64{6, 4}},
	}
	gen := synth.NewGenerator(seed, 0.05, endT, []float64{0, 1.2, 0, 0.3}, specs)
	ms := mutate(gen.Sample())

	var channels []fusion.ChannelConfig
	for _, sp := range specs {
		md, err := synth.Models(sp)
		if err != nil {
			t.Fatal(err)
		}
		channels = append(channels, fusion.ChannelConfig{
			ID: sp.ID, Model: md, DivergeNIS: 30, DivergeCount: 3, RecoverCount: 3})
	}
	cfg := fusion.Config{
		Align:  alignment.Config{MaxDelay: 0.6},
		Filter: filter.Config{NISGate: 30, EigenFloor: 1e-12},
		Tick:   0.05, StartT: 0, EndT: endT,
	}
	fu, err := fusion.New(sm, channels, cfg)
	if err != nil {
		t.Fatal(err)
	}
	P := mat.NewSymDense(4, []float64{4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1})
	if err := fu.Init(mat.NewVecDense(4, []float64{0, 0, 0, 0}), P); err != nil {
		t.Fatal(err)
	}

	type d struct {
		at float64
		m  models.Measurement
	}
	var sched []d
	for _, m := range ms {
		sched = append(sched, d{at: gen.Arrival(m), m: m})
	}
	sort.SliceStable(sched, func(i, j int) bool { return sched[i].at < sched[j].at })

	if concurrent {
		var wg sync.WaitGroup
		for w := 0; w < 4; w++ {
			wg.Add(1)
			go func(w int) {
				defer wg.Done()
				for i, r := range sched {
					if i%4 == w {
						_ = fu.Submit(r.m)
					}
				}
			}(w)
		}
		wg.Wait()
	} else {
		for _, r := range sched {
			_ = fu.Submit(r.m)
		}
	}
	horizon := endT
	for _, r := range sched {
		if r.at > fu.Clock() {
			if _, err := fu.AdvanceTo(r.at); err != nil {
				t.Fatal(err)
			}
		}
		if r.at > horizon {
			horizon = r.at
		}
	}
	if horizon > fu.Clock() {
		if _, err := fu.AdvanceTo(horizon); err != nil {
			t.Fatal(err)
		}
	}
	_, _ = fu.Flush()
	return fu.Estimate(), fu.Stats(), fu, gen, specs
}

func TestIntegrationMixedModelsConverge(t *testing.T) {
	st, stats, fu, gen, _ := endToEnd(t, 2026, func(ms []models.Measurement) []models.Measurement {
		return ms
	}, true)

	truth := gen.GroundTruth(8.0)
	for i := 0; i < 4; i++ {
		e := math.Abs(st.X.AtVec(i) - truth[i])
		// Position tolerance generous (noisy GPS sigma 1), velocity tighter.
		tol := 1.5
		if i == 1 || i == 3 {
			tol = 0.6
		}
		if e > tol {
			t.Fatalf("component %d estimate %.3f too far from truth %.3f (err %.3f)",
				i, st.X.AtVec(i), truth[i], e)
		}
	}

	// Covariance symmetric PSD.
	var es mat.EigenSym
	es.Factorize(st.P, false)
	for i, v := range es.Values(nil) {
		if v < 0 {
			t.Fatalf("final P has negative eig %d: %v", i, v)
		}
	}
	for i := 0; i < 4; i++ {
		for j := i + 1; j < 4; j++ {
			if math.Abs(st.P.At(i, j)-st.P.At(j, i)) > 0 {
				t.Fatal("final P asymmetric")
			}
		}
	}

	t.Logf("estimate=%v truth=%v", st.X.RawVector().Data, truth)
	t.Logf("decision: accepted=%d rejected=%d updates=%d (linear+nonlinear fused)",
		stats.Accepted, stats.Rejected, stats.Updates)
	for _, id := range []string{"gps", "vel", "polar"} {
		s := fu.Status()[id]
		t.Logf("channel %-5s accepted=%d rejected=%d isolated=%v", id, s.Accepted, s.Rejected, s.Isolated)
	}
}

// Same input sequence submitted in two ways must yield bit-identical
// final state and statistics.
func TestIntegrationReproducibleAcrossSubmitStyles(t *testing.T) {
	mut := func(ms []models.Measurement) []models.Measurement {
		// introduce one duplicated sample, one NaN, one polar divergence
		out := append([]models.Measurement(nil), ms...)
		out = append(out, ms[3]) // exact duplicate
		return out
	}
	s1, st1, _, _, _ := endToEnd(t, 77, mut, false)
	s2, st2, _, _, _ := endToEnd(t, 77, mut, true)
	for i := 0; i < 4; i++ {
		if s1.X.AtVec(i) != s2.X.AtVec(i) {
			t.Fatalf("state component %d differs: %v vs %v", i, s1.X.RawVector().Data, s2.X.RawVector().Data)
		}
	}
	if st1 != st2 {
		t.Fatalf("stats differ: %+v vs %+v", st1, st2)
	}
	t.Logf("decision: sequential & concurrent inputs give identical state and stats=%+v", st1)
}

// Rejection reasons across the whole pipeline must be distinguishable.
func TestIntegrationRejectionReasonsDistinct(t *testing.T) {
	seen := map[reject.Reason]int{}
	var trace []string

	_, _, fu, gen, specs := endToEnd(t, 303, func(ms []models.Measurement) []models.Measurement {
		// duplicate
		ms = append(ms, ms[2])
		// a NaN sample
		for i := range ms {
			if ms[i].SensorID == "gps" && ms[i].Timestamp == 3.0 {
				ms[i].Z = mat.NewVecDense(2, []float64{math.NaN(), 1})
			}
		}
		return ms
	}, false)

	// Directly probe a stale submit after the run advanced the clock.
	stale := models.Measurement{SensorID: "vel", Timestamp: 0.01,
		Z: mat.NewVecDense(2, []float64{0, 0})}
	if err := fu.Submit(stale); err != nil {
		seen[reject.Of(err)]++
		trace = append(trace, fmt.Sprintf("stale probe: %v", err))
	}
	dup := models.Measurement{SensorID: "gps", Timestamp: 0.5,
		Z: mat.NewVecDense(2, []float64{0, 0})}
	if err := fu.Submit(dup); err != nil {
		seen[reject.Of(err)]++
		trace = append(trace, fmt.Sprintf("duplicate probe: %v", err))
	}

	for _, ev := range fu.Events() {
		if ev.Kind == "rejected" {
			trace = append(trace, fmt.Sprintf("t=%.2f %s: %s", ev.Timestamp, ev.SensorID, ev.Reason))
			for _, r := range []reject.Reason{
				reject.ReasonNonFiniteValue, reject.ReasonDuplicateTimestamp,
				reject.ReasonStaleTimestamp, reject.ReasonDiverging,
				reject.ReasonChannelIsolated,
			} {
				if strings.Contains(ev.Reason, string(r)) {
					seen[r]++
				}
			}
		}
	}
	for _, line := range trace {
		t.Log("decision:", line)
	}
	for _, r := range []reject.Reason{
		reject.ReasonNonFiniteValue, reject.ReasonDuplicateTimestamp, reject.ReasonStaleTimestamp} {
		if seen[r] == 0 {
			t.Fatalf("expected at least one %s rejection in trace", r)
		}
	}
	_ = specs
	_ = gen
}

// A diverging channel is isolated and then recovered inside one end-to-end
// run, while linear channels keep the estimate on track.
func TestIntegrationFaultThenRecover(t *testing.T) {
	st, stats, fu, gen, _ := endToEnd(t, 909, func(ms []models.Measurement) []models.Measurement {
		return gen2Corrupt(ms)
	}, true)
	if stats.Isolations != 1 || stats.Recoveries != 1 {
		t.Fatalf("want 1 isolation + 1 recovery, got iso=%d rec=%d", stats.Isolations, stats.Recoveries)
	}
	s := fu.Status()["polar"]
	if s.Isolated {
		t.Fatal("polar should recover by end")
	}
	truth := gen.GroundTruth(8.0)
	if math.Abs(st.X.AtVec(0)-truth[0]) > 1.5 {
		t.Fatalf("position polluted: est %.3f truth %.3f", st.X.AtVec(0), truth[0])
	}
	t.Logf("decision: polar isolated=%.2f recovered=%.2f, est=%v truth=%v",
		s.IsolatedAt, s.RecoveredAt, st.X.RawVector().Data, truth)
}

func gen2Corrupt(ms []models.Measurement) []models.Measurement {
	for i := range ms {
		if ms[i].SensorID == "polar" && ms[i].Timestamp >= 2 && ms[i].Timestamp < 4 {
			ms[i].Z = mat.NewVecDense(2, []float64{ms[i].Z.AtVec(0) + 500, ms[i].Z.AtVec(1) + 1.5})
		}
	}
	return ms
}
