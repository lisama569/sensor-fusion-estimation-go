// Command demo runs the local sensor-fusion estimator end to end on
// synthetic multi-rate data, including delayed/out-of-order measurements
// and a temporary single-channel fault, then prints a trace and summary.
//
// Measurements are submitted concurrently by per-channel goroutines at
// their simulated arrival time; the estimator clock is advanced to the
// watermark "minimum arrival time of measurements still in flight", which
// is the standard rule that keeps delayed measurements inside their
// tolerated window.
package main

import (
	"fmt"
	"math"
	"os"
	"sort"
	"sync"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/alignment"
	"github.com/lisama569/sensor-fusion-estimation-go/internal/filter"
	"github.com/lisama569/sensor-fusion-estimation-go/internal/fusion"
	"github.com/lisama569/sensor-fusion-estimation-go/internal/models"
	"github.com/lisama569/sensor-fusion-estimation-go/internal/synth"
	"gonum.org/v1/gonum/mat"
)

type delivery struct {
	at float64
	m  models.Measurement
}

func main() {
	const (
		endT = 10.0
		seed = int64(20260924)
	)

	sm := models.NewConstantVelocity2D(0.5)

	specs := []synth.ChannelSpec{
		{ID: "gps", Kind: "gps", Period: 0.5, Delay: 0.2,
			Noise2D: [2][2]float64{{1.0, 0}, {0, 1.0}}},
		{ID: "vel", Kind: "velocity", Period: 0.25, Delay: 0.05,
			Noise2D: [2][2]float64{{0.25, 0}, {0, 0.25}}},
		{ID: "polar", Kind: "polar", Period: 0.4, Delay: 0.3,
			Noise2D: [2][2]float64{{0.5, 0}, {0, 0.02}}, Beacon: [2]float64{8, 6}},
	}

	gen := synth.NewGenerator(seed, 0.1, endT, []float64{0, 1, 0, 0.5}, specs)
	ms := gen.Sample()
	// Polar channel diverges in a window; GPS gets one NaN sample.
	ms = gen.Corrupt(ms, "polar", 4.0, 6.0, false)
	ms = gen.Corrupt(ms, "gps", 7.0, 7.5, true)

	// Deliveries sorted by arrival time. A shuffle of the per-channel
	// order still happens inside each goroutine; arrival-time ties are
	// resolved deterministically by the alignment buffer.
	var schedule []delivery
	for _, mm := range ms {
		schedule = append(schedule, delivery{at: gen.Arrival(mm), m: mm})
	}
	sort.SliceStable(schedule, func(i, j int) bool { return schedule[i].at < schedule[j].at })

	var channels []fusion.ChannelConfig
	for _, sp := range specs {
		md, err := synth.Models(sp)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		channels = append(channels, fusion.ChannelConfig{
			ID: sp.ID, Model: md,
			DivergeNIS: 20.0, DivergeCount: 3, RecoverCount: 3,
		})
	}

	cfg := fusion.Config{
		Align:  alignment.Config{MaxDelay: 1.0},
		Filter: filter.Config{NISGate: 20.0, EigenFloor: 1e-12},
		Tick:   0.1, StartT: 0, EndT: endT,
	}
	fu, err := fusion.New(sm, channels, cfg)
	must(err)

	P0 := mat.NewSymDense(4, []float64{
		4, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 4, 0,
		0, 0, 0, 1,
	})
	must(fu.Init(mat.NewVecDense(4, []float64{0, 0, 0, 0}), P0))

	// Phase 1: concurrent producers submit ALL measurements. Submission
	// order between goroutines is scheduling-dependent (real lock
	// contention, real races against -race), but while the clock is
	// untouched the watermark is -Inf so nothing can be judged stale.
	var wg sync.WaitGroup
	for _, sp := range specs {
		sp := sp
		var batch []delivery
		for _, d := range schedule {
			if d.m.SensorID == sp.ID {
				batch = append(batch, d)
			}
		}
		wg.Add(1)
		go func(batch []delivery) {
			defer wg.Done()
			for _, d := range batch {
				_ = fu.Submit(d.m)
			}
		}(batch)
	}
	wg.Wait()

	// Phase 2: single-threaded deterministic watermark driver. The clock
	// only advances along observed arrival times, so every delayed
	// measurement is released in (timestamp, sensorID) order regardless of
	// the order in which phase 1 happened to enqueue it.
	for _, d := range schedule {
		if d.at > fu.Clock() {
			_, _ = fu.AdvanceTo(d.at)
		}
	}
	if endT > fu.Clock() {
		_, _ = fu.AdvanceTo(endT)
	}
	if _, err := fu.Flush(); err != nil {
		// ReasonNoData after a full run is expected and harmless.
		_ = err
	}

	est := fu.Estimate()
	truth := gen.GroundTruth(endT)
	fmt.Println("== sensor fusion demo ==")
	fmt.Printf("final time: %.3f\n", est.T)
	fmt.Printf("estimate  : [x=%8.4f vx=%8.4f y=%8.4f vy=%8.4f]\n",
		est.X.AtVec(0), est.X.AtVec(1), est.X.AtVec(2), est.X.AtVec(3))
	fmt.Printf("truth     : [x=%8.4f vx=%8.4f y=%8.4f vy=%8.4f]\n",
		truth[0], truth[1], truth[2], truth[3])
	fmt.Printf("error     : [%.4f %.4f %.4f %.4f]\n",
		est.X.AtVec(0)-truth[0], est.X.AtVec(1)-truth[1],
		est.X.AtVec(2)-truth[2], est.X.AtVec(3)-truth[3])

	var es mat.EigenSym
	es.Factorize(est.P, false)
	vals := es.Values(nil)
	minE, maxE := vals[0], vals[0]
	for _, v := range vals {
		minE = math.Min(minE, v)
		maxE = math.Max(maxE, v)
	}
	symErr := 0.0
	for i := 0; i < 4; i++ {
		for j := i + 1; j < 4; j++ {
			symErr = math.Max(symErr, math.Abs(est.P.At(i, j)-est.P.At(j, i)))
		}
	}
	fmt.Printf("P eig min/max = %.3e / %.3e, asymmetry = %.3e\n", minE, maxE, symErr)

	st := fu.Stats()
	fmt.Printf("stats: accepted=%d rejected=%d nonfinite=%d degenerate=%d diverging=%d isolated=%d recovered=%d duplicates=%d stale=%d\n",
		st.Accepted, st.Rejected, st.NonFinite, st.Degenerate, st.Diverging,
		st.Isolations, st.Recoveries, st.Duplicates, st.Stale)
	for _, id := range []string{"gps", "vel", "polar"} {
		s := fu.Status()[id]
		fmt.Printf("channel %-5s isolated=%-5v accepted=%3d rejected=%3d isolatedAt=%.2f recoveredAt=%.2f\n",
			id, s.Isolated, s.Accepted, s.Rejected, s.IsolatedAt, s.RecoveredAt)
	}

	fmt.Println("-- non-queued decision trace --")
	for _, ev := range fu.Events() {
		if ev.Kind == "queued" || ev.Kind == "predicted" {
			continue
		}
		nis := ""
		if ev.NIS != 0 {
			nis = fmt.Sprintf(" nis=%.2f", ev.NIS)
		}
		reason := ev.Reason
		if reason == "" {
			reason = "-"
		}
		fmt.Printf("t=%6.2f %-9s sensor=%-5s reason=%s%s\n", ev.Timestamp, ev.Kind, ev.SensorID, reason, nis)
	}
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "fatal:", err)
		os.Exit(1)
	}
}
