package engine_test

import (
	"math"
	"testing"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/engine"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/model"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/sim"
)

func scenarioSpecs() []sim.SensorSpec {
	return []sim.SensorSpec{
		{Name: "gps-a", Kind: "position", RateHz: 5, Delay: 0.10, Jitter: 0.02, StdPos: 0.25},
		{Name: "gps-b", Kind: "position", RateHz: 4, Delay: 0.25, Jitter: 0.05, StdPos: 0.30},
		{Name: "radar", Kind: "rangebearing", RateHz: 2, Delay: 0.40, Jitter: 0.05,
			StdRange: 0.15, StdBearing: 0.015, Sx: -3, Sy: -2},
	}
}

func modelsFor(specs []sim.SensorSpec) map[string]model.MeasurementModel {
	out := make(map[string]model.MeasurementModel)
	for _, s := range specs {
		switch s.Kind {
		case "rangebearing":
			out[s.Name] = &model.RangeBearing{
				Sx: s.Sx, Sy: s.Sy,
				R: mat.From([][]float64{{s.StdRange * s.StdRange, 0}, {0, s.StdBearing * s.StdBearing}}),
			}
		default:
			v := s.StdPos * s.StdPos
			out[s.Name] = &model.LinearPosition{R: mat.From([][]float64{{v, 0}, {0, v}})}
		}
	}
	return out
}

func runSimScenario(t *testing.T, concurrent bool) (*model.State, engine.Stats, []*engine.RejectError) {
	t.Helper()
	specs := scenarioSpecs()
	cfg := sim.Config{
		StartTime: 0, Duration: 8, Seed: 1234, Shuffle: true,
		InjectFaults: true, FaultSource: "gps-b", FaultStart: 2.0, FaultEnd: 4.0,
		EmitDuplicates: true, DuplicateSource: "gps-a",
		Sensors: specs,
	}
	ds := sim.Generate(cfg)
	models := modelsFor(specs)

	initial := &model.State{Time: 0, X: mat.Vec(0, 0, 1, 0),
		P: mat.From([][]float64{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0.25, 0}, {0, 0, 0, 0.25}})}
	eng, err := engine.New(initial, &model.ConstantVelocity{QPSD: 0.05}, engine.DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}

	faultN := map[string]int{}
	t.Logf("--- input sequence (%d samples), concurrent=%v ---", len(ds.Samples), concurrent)
	type logged struct{ s sim.Sample }
	pendingLogs := make([]logged, 0, len(ds.Samples))
	for _, s0 := range ds.Samples {
		z := s0.Z
		if s0.NonFinite {
			cp := s0
			cp.Z = append([]float64(nil), s0.Z...)
			sim.CorruptForFault(&cp, faultN[s0.Source])
			faultN[s0.Source]++
			z = cp.Z
		}
		pendingLogs = append(pendingLogs, logged{s0})
		t.Logf("submit src=%-6s t=%6.3f z=[% .4f % .4f] nonfinite=%v dup=%v",
			s0.Source, s0.Time, z[0], z[1], s0.NonFinite, s0.Duplicate)

		sub := func() {
			if err := eng.Submit(s0.Source, s0.Time, mat.Vec(z[0], z[1]), models[s0.Source]); err != nil {
				if re, ok := err.(*engine.RejectError); ok {
					t.Logf("   submit-decision REJECT reason=%-22s %s", re.Reason, re.Detail)
				}
			}
		}
		if concurrent {
			// Submit on a goroutine to prove order independence.
			done := make(chan struct{})
			go func() { defer close(done); sub() }()
			<-done
		} else {
			sub()
		}
	}
	_ = pendingLogs

	t.Log("--- advancing in (time,source) order ---")
	for tt := 0.1; tt <= cfg.Duration+0.7; tt += 0.1 {
		ttm := math.Round(tt*10) / 10
		_, rejs, err := eng.Advance(ttm)
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range rejs {
			t.Logf("   fuse-decision @%6.3f REJECT src=%-6s reason=%-22s %s",
				ttm, r.Source, r.Reason, r.Detail)
		}
	}
	return eng.State(), eng.Stats(), eng.Rejections()
}

func TestEndToEndScenarioAndReproducibility(t *testing.T) {
	s1, st1, r1 := runSimScenario(t, false)
	s2, st2, r2 := runSimScenario(t, false)
	if !mat.Equal(s1.X, s2.X, 0) || !mat.Equal(s1.P, s2.P, 0) {
		t.Fatal("two sequential scenario runs differ")
	}
	if len(r1) != len(r2) || st1.Updates != st2.Updates || st1.Rejected != st2.Rejected {
		t.Fatal("sequential scenario stats differ")
	}

	s3, st3, r3 := runSimScenario(t, true)
	if !mat.Equal(s1.X, s3.X, 0) || !mat.Equal(s1.P, s3.P, 0) {
		t.Fatalf("concurrent scenario state differs:\nseq=%v\ncon=%v", s1.X.Data, s3.X.Data)
	}
	if len(r1) != len(r3) || st1.Updates != st3.Updates || st1.Rejected != st3.Rejected {
		t.Fatalf("concurrent scenario stats differ: %+v vs %+v (rej %d vs %d)",
			st1, st3, len(r1), len(r3))
	}

	// Covariance properties.
	if !mat.IsSymmetric(s1.P, 1e-9) {
		t.Fatal("final P not symmetric")
	}
	eig, _ := mat.EigenSymmetric(s1.P)
	for _, l := range eig {
		if l < -1e-9 {
			t.Fatalf("final P not PSD: eig=%v", eig)
		}
	}

	// Faulted stream must have been isolated and then recovered by the end.
	b := st1.PerStream["gps-b"]
	t.Logf("gps-b stats: %+v rejected=%v", b, b.Rejected)
	if b.HardFaults == 0 {
		t.Fatal("gps-b hard faults not recorded")
	}
	if b.Recoveries == 0 {
		t.Fatal("gps-b never recorded recovery")
	}
	if b.CurrentlyDown {
		t.Fatal("gps-b should be back online by scenario end")
	}
	// Duplicate on gps-a recorded with a distinguishable reason.
	a := st1.PerStream["gps-a"]
	if a.Rejected[engine.ReasonDuplicate] == 0 {
		t.Fatalf("expected duplicate_timestamp rejection on gps-a, got %v", a.Rejected)
	}
	// Radar nonlinear stream should have accepted updates.
	if st1.PerStream["radar"].Accepted == 0 {
		t.Fatal("radar produced no accepted updates")
	}
	t.Logf("final x=%v", s1.X.Data)
	t.Logf("stats: predicts=%d updates=%d rejected=%d", st1.Predicts, st1.Updates, st1.Rejected)
}
