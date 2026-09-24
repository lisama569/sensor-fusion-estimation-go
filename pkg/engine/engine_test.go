package engine

import (
	"fmt"
	"math"
	"sync"
	"testing"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/model"
)

// buildScenario creates a fixed multi-rate dataset with a duplicate, an
// out-of-order late sample and a non-finite fault, returned in the given
// emission order.
type sample struct {
	src string
	t   float64
	z   *mat.Vector
}

func scenarioModels() map[string]model.MeasurementModel {
	return map[string]model.MeasurementModel{
		"a": posModel(0.2),
		"b": posModel(0.3),
		"r": &model.RangeBearing{Sx: -2, Sy: -2, R: mat.From([][]float64{{0.04, 0}, {0, 0.0004}})},
	}
}

func buildSequence() []sample {
	var s []sample
	// regular 5 Hz position A
	for k := 0; k <= 20; k++ {
		tt := float64(k) * 0.2
		s = append(s, sample{"a", tt, mat.Vec(tt, 0.1*math.Sin(tt))})
	}
	// 4 Hz position B, with a non-finite fault around t=1.5-2.5
	for k := 0; k <= 16; k++ {
		tt := float64(k) * 0.25
		z := mat.Vec(tt, 0)
		if tt >= 1.5 && tt < 2.5 {
			if k%2 == 0 {
				z = mat.Vec(math.NaN(), 0)
			} else {
				z = mat.Vec(math.Inf(1), 0)
			}
		}
		s = append(s, sample{"b", tt, z})
	}
	// 2 Hz radar nonlinear
	for k := 0; k <= 8; k++ {
		tt := float64(k) * 0.5
		px, py := tt, 0.0
		s = append(s, sample{"r", tt, mat.Vec(math.Hypot(px+2, py+2), math.Atan2(py+2, px+2))})
	}
	// explicit duplicate on a (already has 0.4)
	s = append(s, sample{"a", 0.4, mat.Vec(9, 9)})
	// explicit expired sample: a timestamp far older than the committed
	// clock (default horizon 2s) so it must be refused as expired.
	s = append(s, sample{"a", -5.0, mat.Vec(0, 0)})
	return s
}

func runSequence(t *testing.T, seq []sample, concurrent bool) (*model.State, Stats, []*RejectError) {
	eng, err := New(initialState(), &model.ConstantVelocity{QPSD: 0.05}, DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	models := scenarioModels()
	submit := func(sm sample) {
		_ = eng.Submit(sm.src, sm.t, sm.z, models[sm.src])
	}
	if concurrent {
		var wg sync.WaitGroup
		for _, sm := range seq {
			wg.Add(1)
			go func(sm sample) {
				defer wg.Done()
				submit(sm)
			}(sm)
		}
		wg.Wait()
	} else {
		for _, sm := range seq {
			submit(sm)
		}
	}
	for tt := 0.1; tt <= 4.5; tt += 0.1 {
		if _, _, err := eng.Advance(math.Round(tt*10) / 10); err != nil {
			t.Fatal(err)
		}
	}
	return eng.State(), eng.Stats(), eng.Rejections()
}

// TestReproducibilityConcurrentVsSequential requires the concurrent and
// sequential ingestion of the identical sequence to yield bit-identical
// final state, covariance and rejection counts.
func TestReproducibilityConcurrentVsSequential(t *testing.T) {
	seq := buildSequence()

	var st1 *model.State
	var stats1 Stats
	var rej1 []*RejectError
	var st2 *model.State
	var stats2 Stats
	var rej2 []*RejectError

	// Two sequential runs and one concurrent run must all agree.
	st1, stats1, rej1 = runSequence(t, seq, false)
	st2, stats2, rej2 = runSequence(t, seq, false)
	if !mat.Equal(st1.X, st2.X, 0) || !mat.Equal(st1.P, st2.P, 0) {
		t.Fatal("sequential runs not bit-identical")
	}
	if fmt.Sprint(stats1) != fmt.Sprint(stats2) || len(rej1) != len(rej2) {
		t.Fatal("sequential stats/rejections differ")
	}

	st3, stats3, rej3 := runSequence(t, seq, true)
	if !mat.Equal(st1.X, st3.X, 0) || !mat.Equal(st1.P, st3.P, 0) {
		t.Fatalf("concurrent final state differs from sequential\nseq=%v\ncon=%v",
			st1.X.Data, st3.X.Data)
	}
	if stats1.Updates != stats3.Updates || stats1.Rejected != stats3.Rejected ||
		stats1.Predicts != stats3.Predicts || len(rej1) != len(rej3) {
		t.Fatalf("concurrent stats differ: %+v vs %+v (rej %d vs %d)",
			stats1, stats3, len(rej1), len(rej3))
	}
	assertPSD(t, st3.P, "concurrent P")
}

// TestRejectionReasonsDistinguishable asserts every required rejection class
// appears at least once and the typed reasons are distinct.
func TestRejectionReasonsDistinguishable(t *testing.T) {
	seq := buildSequence()
	_, stats, rejs := runSequence(t, seq, false)
	counts := map[RejectReason]int{}
	for _, r := range rejs {
		counts[r.Reason]++
	}
	required := []RejectReason{
		ReasonNonFinite,
		ReasonDuplicate,
		ReasonExpired,
		ReasonStreamFaulty,
	}
	t.Logf("rejection counts: %v", counts)
	for _, rr := range required {
		if counts[rr] == 0 {
			t.Fatalf("expected at least one %q rejection, got counts=%v", rr, counts)
		}
	}
	if stats.PerStream["b"].Recoveries < 1 {
		t.Fatalf("stream b should have recovered at least once: %+v", stats.PerStream["b"])
	}
	if !stats.PerStream["b"].CurrentlyDown == false && false {
		t.Fatal("unreachable")
	}
	if stats.PerStream["b"].CurrentlyDown {
		t.Fatal("stream b should be back online by end of scenario")
	}
}

// TestIllegalNoiseThroughEngine ensures an illegal R is surfaced with a
// distinguishable reason and does not abort the pipeline.
func TestIllegalNoiseThroughEngine(t *testing.T) {
	eng, _ := New(initialState(), &model.ConstantVelocity{QPSD: 0.05}, DefaultConfig())
	bad := &model.LinearPosition{R: mat.From([][]float64{{1, 0}, {0, -0.5}})}
	if err := eng.Submit("x", 0.1, mat.Vec(0, 0), bad); err != nil {
		t.Fatalf("submit should buffer; got %v", err)
	}
	_, rejs, err := eng.Advance(0.1)
	if err != nil {
		t.Fatal(err)
	}
	if len(rejs) != 1 || rejs[0].Reason != ReasonIllegalNoise {
		t.Fatalf("want illegal_noise rejection, got %v", rejs)
	}
	// A legal measurement on the same stream still works afterward.
	good := posModel(0.1)
	if err := eng.Submit("x", 0.2, mat.Vec(0.2, 0), good); err != nil {
		t.Fatal(err)
	}
	_, rejs2, _ := eng.Advance(0.2)
	if len(rejs2) != 0 {
		t.Fatalf("legal measurement after illegal noise rejected: %v", rejs2)
	}
}

// TestConcurrentDeterministicSameMultiset submits the identical multiset of
// measurements from many goroutines, joins all of them, and only then drives
// the fixed clock. The result must be bit-identical to a sequential ingest:
// concurrency changes arrival scheduling but not the (time, source) aligned
// outcome. This is the determinism contract.
func TestConcurrentDeterministicSameMultiset(t *testing.T) {
	models := scenarioModels()
	seqIn := buildSequence()

	run := func(concurrent bool) []float64 {
		eng, _ := New(initialState(), &model.ConstantVelocity{QPSD: 0.05}, DefaultConfig())
		if concurrent {
			var wg sync.WaitGroup
			for _, sm := range seqIn {
				wg.Add(1)
				go func(sm sample) {
					defer wg.Done()
					_ = eng.Submit(sm.src, sm.t, sm.z, models[sm.src])
				}(sm)
			}
			wg.Wait()
		} else {
			for _, sm := range seqIn {
				_ = eng.Submit(sm.src, sm.t, sm.z, models[sm.src])
			}
		}
		for tt := 0.1; tt <= 4.5; tt += 0.1 {
			if _, _, err := eng.Advance(math.Round(tt*10) / 10); err != nil {
				t.Fatal(err)
			}
		}
		st := eng.State()
		return append(append([]float64{}, st.X.Data...), st.P.Data...)
	}

	base := run(false)
	for i := 0; i < 8; i++ {
		con := run(true)
		for j := range base {
			if base[j] != con[j] {
				t.Fatalf("run %d differs at %d: %.18f vs %.18f", i, j, base[j], con[j])
			}
		}
	}
}

// TestConcurrentAPISafety hammers Submit/Advance/State/Stats simultaneously
// solely to be validated by the race detector; final covariance must remain
// symmetric PSD and finite regardless of interleaving.
func TestConcurrentAPISafety(t *testing.T) {
	eng, _ := New(initialState(), &model.ConstantVelocity{QPSD: 0.05}, DefaultConfig())
	models := scenarioModels()
	var wg sync.WaitGroup
	stop := make(chan struct{})
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			k := 0
			for {
				select {
				case <-stop:
					return
				default:
				}
				tt := float64(k%40) * 0.1
				src := []string{"a", "b", "r"}[g%3]
				_ = eng.Submit(src, tt, mat.Vec(tt, 0), models[src])
				_, _, _ = eng.Advance(tt)
				_ = eng.State()
				_ = eng.Stats()
				k++
				if k > 100 {
					return
				}
			}
		}(g)
	}
	wg.Wait()
	close(stop)
	assertPSD(t, eng.State().P, "P after contention")
}
