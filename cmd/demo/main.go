// Command demo runs the local sensor-fusion pipeline on a deterministic
// simulated multi-sensor scenario and prints the input sequence, per-sample
// decisions (fused / rejected with reason) and the final estimate.
package main

import (
	"fmt"
	"math"
	"os"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/engine"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/model"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/sim"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "demo failed:", err)
		os.Exit(1)
	}
}

func sensorSpecs() []sim.SensorSpec {
	return []sim.SensorSpec{
		{Name: "gps-a", Kind: "position", RateHz: 5, Delay: 0.10, Jitter: 0.03, StdPos: 0.25},
		{Name: "gps-b", Kind: "position", RateHz: 4, Delay: 0.25, Jitter: 0.05, StdPos: 0.35},
		{Name: "radar", Kind: "rangebearing", RateHz: 2, Delay: 0.40, Jitter: 0.08,
			StdRange: 0.20, StdBearing: 0.02, Sx: -3, Sy: -2},
	}
}

func buildModel(s sim.SensorSpec) model.MeasurementModel {
	switch s.Kind {
	case "rangebearing":
		return &model.RangeBearing{
			Sx: s.Sx, Sy: s.Sy,
			R: mat.From([][]float64{
				{s.StdRange * s.StdRange, 0},
				{0, s.StdBearing * s.StdBearing},
			}),
		}
	default:
		v := s.StdPos * s.StdPos
		return &model.LinearPosition{R: mat.From([][]float64{{v, 0}, {0, v}})}
	}
}

func run() error {
	specs := sensorSpecs()
	models := make(map[string]model.MeasurementModel, len(specs))
	for _, s := range specs {
		models[s.Name] = buildModel(s)
	}

	cfg := sim.Config{
		StartTime:       0,
		Duration:        8,
		Seed:            42,
		Shuffle:         true,
		InjectFaults:    true,
		FaultSource:     "gps-b",
		FaultStart:      2.0,
		FaultEnd:        4.5,
		EmitDuplicates:  true,
		DuplicateSource: "gps-a",
		Sensors:         specs,
	}
	ds := sim.Generate(cfg)

	initial := &model.State{
		Time: 0,
		X:    mat.Vec(0, 0, 1, 0),
		P: mat.From([][]float64{
			{1, 0, 0, 0},
			{0, 1, 0, 0},
			{0, 0, 0.25, 0},
			{0, 0, 0, 0.25},
		}),
	}
	motion := &model.ConstantVelocity{QPSD: 0.05}
	eng, err := engine.New(initial, motion, engine.DefaultConfig())
	if err != nil {
		return err
	}

	fmt.Println("=== input sequence (emission order, shuffled/delayed) ===")
	faultIdx := map[string]int{}
	for i, s := range ds.Samples {
		z := s.Z
		tag := ""
		if s.NonFinite {
			cp := sim.Sample{Time: s.Time, Source: s.Source, Z: append([]float64(nil), s.Z...)}
			sim.CorruptForFault(&cp, faultIdx[s.Source])
			faultIdx[s.Source]++
			z = cp.Z
			tag = " [INJECTED NON-FINITE]"
		}
		if s.Duplicate {
			tag += " [DUPLICATE]"
		}
		fmt.Printf("in  #%-3d src=%-6s t=%6.3f z=[% .4f % .4f]%s\n",
			i, s.Source, s.Time, z[0], z[1], tag)

		if err := eng.Submit(s.Source, s.Time, mat.Vec(z[0], z[1]), models[s.Source]); err != nil {
			fmt.Printf("     -> SUBMIT REJECTED: %v\n", err)
		}
	}

	fmt.Println("\n=== advancing time, fusing in (time,source) order ===")
	const step = 0.1
	stop := cfg.Duration + 0.6
	for tt := 0.1; tt <= stop+1e-9; tt += step {
		_, rejs, err := eng.Advance(round1(tt))
		if err != nil {
			return err
		}
		for _, r := range rejs {
			fmt.Printf("  FUSE REJECTED @t=%6.3f src=%-6s reason=%-24s %s\n",
				tt, r.Source, r.Reason, r.Detail)
		}
	}

	st := eng.State()
	last := ds.Truth[len(ds.Truth)-1]
	fmt.Println("\n=== final estimate ===")
	fmt.Printf("time   = %.3f\n", st.Time)
	fmt.Printf("x      = [%.4f %.4f %.4f %.4f]\n",
		mat.VecAt(st.X, 0), mat.VecAt(st.X, 1), mat.VecAt(st.X, 2), mat.VecAt(st.X, 3))
	fmt.Printf("truth  = [%.4f %.4f %.4f %.4f]\n", last.Px, last.Py, last.Vx, last.Vy)
	fmt.Printf("pos err= %.4f m\n", math.Hypot(mat.VecAt(st.X, 0)-last.Px, mat.VecAt(st.X, 1)-last.Py))
	fmt.Println("P=")
	for r := 0; r < 4; r++ {
		fmt.Printf("  [%.5f %.5f %.5f %.5f]\n",
			st.P.At(r, 0), st.P.At(r, 1), st.P.At(r, 2), st.P.At(r, 3))
	}

	fmt.Println("\n=== statistics ===")
	st2 := eng.Stats()
	fmt.Printf("predicts=%d updates=%d totalRejected=%d\n", st2.Predicts, st2.Updates, st2.Rejected)
	for name, ss := range st2.PerStream {
		fmt.Printf("stream %-6s accepted=%d hard=%d soft=%d recovered=%d down=%v rejected=%v\n",
			name, ss.Accepted, ss.HardFaults, ss.SoftFaults, ss.Recoveries, ss.CurrentlyDown, ss.Rejected)
	}
	return nil
}

func round1(v float64) float64 { return math.Round(v*10) / 10 }
