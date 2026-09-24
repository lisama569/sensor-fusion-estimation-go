package filter

import (
	"math"
	"testing"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/models"
	"github.com/lisama569/sensor-fusion-estimation-go/internal/reject"
	"gonum.org/v1/gonum/mat"
)

func symEig(t *testing.T, P *mat.SymDense) []float64 {
	t.Helper()
	n, _ := P.Dims()
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if math.Abs(P.At(i, j)-P.At(j, i)) > 1e-12 {
				t.Fatalf("P asymmetric at (%d,%d): %v vs %v", i, j, P.At(i, j), P.At(j, i))
			}
		}
	}
	var es mat.EigenSym
	es.Factorize(P, false)
	return es.Values(nil)
}

func newGPSModel(t *testing.T, Rdiag float64) *models.LinearModel {
	H := mat.NewDense(2, 4, []float64{
		1, 0, 0, 0,
		0, 0, 1, 0,
	})
	R := mat.NewSymDense(2, []float64{Rdiag, 0, 0, Rdiag})
	m, err := models.NewLinearModel(4, H, nil, R)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func initFilter(t *testing.T, cfg Config) *Filter {
	sm := models.NewConstantVelocity2D(0.5)
	f := New(sm, cfg)
	P := mat.NewSymDense(4, []float64{
		4, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 4, 0,
		0, 0, 0, 1,
	})
	if err := f.Init(0, mat.NewVecDense(4, []float64{0, 0, 0, 0}), P); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestPredictAndCovariancePSD(t *testing.T) {
	f := initFilter(t, DefaultConfig())
	steps := []float64{0.1, 0.1, 0.25, 0.5, 1.0, 2.0}
	for _, dt := range steps {
		target := f.State().T + dt
		if err := f.PredictTo(target); err != nil {
			t.Fatalf("predict %.2f: %v", dt, err)
		}
		vals := symEig(t, f.State().P)
		for i, v := range vals {
			if v < -1e-10 {
				t.Fatalf("after predict P not PSD, eig[%d]=%v", i, v)
			}
		}
	}
	t.Logf("decision: P stays symmetric PSD through %v; eig=%v", steps, symEig(t, f.State().P))
}

func TestIllegalPredictBackwards(t *testing.T) {
	f := initFilter(t, DefaultConfig())
	if err := f.PredictTo(1.0); err != nil {
		t.Fatal(err)
	}
	err := f.PredictTo(0.5)
	if err == nil || reject.Of(err) != reject.ReasonStaleTimestamp {
		t.Fatalf("backwards predict must be rejected, got %v", err)
	}
	t.Logf("decision: predict 1.0 -> 0.5 rejected: %v", err)
}

func TestLinearUpdateReducesUncertaintyAndStaysPSD(t *testing.T) {
	f := initFilter(t, DefaultConfig())
	before := symEig(t, f.State().P)

	if err := f.PredictTo(0.5); err != nil {
		t.Fatal(err)
	}
	m := newGPSModel(t, 1.0)
	mm := models.Measurement{SensorID: "gps", Timestamp: 0.5,
		Z: mat.NewVecDense(2, []float64{0.5, 0.25})}
	out, err := f.Update("gps", mm, m)
	if err != nil || !out.Accepted {
		t.Fatalf("update: accepted=%v err=%v", out.Accepted, err)
	}
	after := symEig(t, f.State().P)
	for i := range after {
		if after[i] < 0 {
			t.Fatalf("post-update P negative eig: %v", after)
		}
	}
	// Position uncertainty should shrink after a direct position fix.
	if f.State().P.At(0, 0) >= 4.0+1e-9 {
		t.Fatalf("position variance did not shrink: %v", f.State().P.At(0, 0))
	}
	t.Logf("decision: GPS update accepted nis=%.3f; eig before=%v after=%v", out.NIS, before, after)
}

// A zero innovation covariance (R=0 with an already exact H direction) is
// singular only when HPH' is also singular; here P starts positive so S is
// invertible. We force degeneracy by updating twice with R=0 so the
// observed direction variance collapses to zero, then a third identical
// update must be rejected as degenerate rather than producing NaNs.
func TestDegenerateUpdateRejected(t *testing.T) {
	cfg := Config{EigenFloor: 0} // do not refloor P so the direction truly collapses
	f := initFilter(t, cfg)
	H := mat.NewDense(1, 4, []float64{1, 0, 0, 0})
	R := mat.NewSymDense(1, []float64{0})
	m, err := models.NewLinearModel(4, H, nil, R)
	if err != nil {
		t.Fatal(err)
	}
	z := func(v float64) models.Measurement {
		return models.Measurement{SensorID: "x", Timestamp: 0.1, Z: mat.NewVecDense(1, []float64{v})}
	}
	if err := f.PredictTo(0.1); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Update("x", z(0.1), m); err != nil {
		t.Fatalf("first exact update: %v", err)
	}
	t.Logf("decision: first exact R=0 update accepted; Pxx=%.3e", f.State().P.At(0, 0))
	_, err = f.Update("x", z(0.1), m)
	if err == nil {
		t.Fatal("singular S update must be rejected")
	}
	if reject.Of(err) != reject.ReasonDegenerateUpdate {
		t.Fatalf("want %s, got %v", reject.ReasonDegenerateUpdate, err)
	}
	t.Logf("decision: second exact update rejected (S singular): %v", err)

	// The prior state must be untouched and still finite/PSD.
	for i := 0; i < 4; i++ {
		if math.IsNaN(f.State().X.AtVec(i)) {
			t.Fatal("state contaminated by degenerate update")
		}
	}
}

func TestNonFiniteMeasurementRejected(t *testing.T) {
	f := initFilter(t, DefaultConfig())
	m := newGPSModel(t, 1)
	mm := models.Measurement{SensorID: "gps", Timestamp: 0.1,
		Z: mat.NewVecDense(2, []float64{math.NaN(), 0})}
	_ = f.PredictTo(0.1)
	_, err := f.Update("gps", mm, m)
	if err == nil || reject.Of(err) != reject.ReasonNonFiniteInnovation {
		// NaN flows into innovation -> ReasonNonFiniteInnovation expected
		t.Fatalf("want non_finite_innovation, got %v", err)
	}
	t.Logf("decision: NaN measurement rejected: %v", err)
}

func TestDimensionMismatchRejected(t *testing.T) {
	f := initFilter(t, DefaultConfig())
	m := newGPSModel(t, 1)
	mm := models.Measurement{SensorID: "gps", Timestamp: 0.1,
		Z: mat.NewVecDense(3, []float64{0, 0, 0})}
	_ = f.PredictTo(0.1)
	_, err := f.Update("gps", mm, m)
	if err == nil || reject.Of(err) != reject.ReasonDimensionMismatch {
		t.Fatalf("want dimension_mismatch, got %v", err)
	}
	t.Logf("decision: 3-vector for 2D model rejected: %v", err)
}

// Nonlinear range-bearing model: bearing residual must be wrapped across
// the +/-pi cut, so a true target just "before" -pi and a measurement just
// "after" +pi are treated as a tiny residual, not a huge one.
func TestNonlinearBearingWrap(t *testing.T) {
	sm := models.NewConstantVelocity2D(0.0)
	f := New(sm, Config{EigenFloor: 1e-12})
	P := mat.NewSymDense(4, []float64{
		1e-6, 0, 0, 0,
		0, 1e-6, 0, 0,
		0, 0, 1e-6, 0,
		0, 0, 0, 1e-6,
	})
	// Platform at (-eps, 1) relative to beacon at origin: range exactly 1,
	// bearing pi/2+eps (just past +pi/2). We instead exercise the +/-pi
	// cut by placing it at (-1, eps) and reporting range exactly 1 with
	// bearing from the other side of the cut; range residual is zero to
	// isolate the angle-wrap behavior.
	eps := 0.01
	if err := f.Init(0, mat.NewVecDense(4, []float64{-1, 0, eps, 0}), P); err != nil {
		t.Fatal(err)
	}
	_ = eps
	bearingModel := func(t *testing.T) *models.NonlinearModel {
		eval := func(x *mat.VecDense) *mat.VecDense {
			return mat.NewVecDense(2, []float64{math.Hypot(x.AtVec(0), x.AtVec(2)), math.Atan2(x.AtVec(2), x.AtVec(0))})
		}
		jac := func(x *mat.VecDense) *mat.Dense {
			r := math.Hypot(x.AtVec(0), x.AtVec(2))
			if r < 1e-9 {
				r = 1e-9
			}
			return mat.NewDense(2, 4, []float64{
				x.AtVec(0) / r, 0, x.AtVec(2) / r, 0,
				-x.AtVec(2) / (r * r), 0, x.AtVec(0) / (r * r), 0,
			})
		}
		R := mat.NewSymDense(2, []float64{1e-2, 0, 0, 1e-2})
		m, err := models.NewNonlinearModel(4, 2, eval, jac, nil, R, 1)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}(t)

	// True bearing atan2(eps,-1) = pi-eps. Report the identical direction
	// from the other side of the cut as -pi+eps and range 1 (matching the
	// Jacobian's linearization range); the wrapped residual must be ~0,
	// not ~2*pi.
	measBearing := -math.Pi + eps
	mm := models.Measurement{SensorID: "polar", Timestamp: 0.1,
		Z: mat.NewVecDense(2, []float64{1.0, measBearing})}
	if err := f.PredictTo(0.1); err != nil {
		t.Fatal(err)
	}
	out, err := f.Update("polar", mm, bearingModel)
	if err != nil || !out.Accepted {
		t.Fatalf("polar update: %v %v", out, err)
	}
	if out.NIS > 5.0 {
		t.Fatalf("angle wrap failed, NIS too large: %.6f", out.NIS)
	}
	t.Logf("decision: bearing measured at %.4f vs true pi wrapped, nis=%.4f", measBearing, out.NIS)
}

// Reproducibility: identical sequence through two independent filters gives
// identical states.
func TestReproducible(t *testing.T) {
	run := func() []float64 {
		f := initFilter(t, DefaultConfig())
		m := newGPSModel(t, 0.5)
		for i, ts := range []float64{0.5, 1.0, 1.5} {
			if err := f.PredictTo(ts); err != nil {
				t.Fatal(err)
			}
			z := mat.NewVecDense(2, []float64{float64(i) * 0.5, 0})
			if _, err := f.Update("gps",
				models.Measurement{SensorID: "gps", Timestamp: ts, Z: z}, m); err != nil {
				t.Fatal(err)
			}
		}
		st := f.State()
		return []float64{st.T, st.X.AtVec(0), st.X.AtVec(1), st.X.AtVec(2), st.X.AtVec(3),
			st.P.At(0, 0), st.P.At(2, 2)}
	}
	a, b := run(), run()
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("non-reproducible at %d: %v vs %v", i, a, b)
		}
	}
	t.Logf("decision: two independent runs identical: %v", a)
}
