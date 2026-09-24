package engine

import (
	"math"
	"testing"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/model"
)

func diag2(a, b float64) *mat.Matrix { return mat.From([][]float64{{a, 0}, {0, b}}) }

func initialState() *model.State {
	return &model.State{
		Time: 0,
		X:    mat.Vec(0, 0, 1, 0),
		P: mat.From([][]float64{
			{1, 0, 0, 0},
			{0, 1, 0, 0},
			{0, 0, 0.25, 0},
			{0, 0, 0, 0.25},
		}),
	}
}

func assertPSD(t *testing.T, m *mat.Matrix, label string) {
	t.Helper()
	if !mat.IsSymmetric(m, 1e-8) {
		t.Fatalf("%s not symmetric", label)
	}
	eig, err := mat.EigenSymmetric(m)
	if err != nil {
		t.Fatalf("%s eigen error: %v", label, err)
	}
	for _, l := range eig {
		if l < -1e-8 {
			t.Fatalf("%s not PSD, eigenvalue %.6e (all=%v)", label, l, eig)
		}
	}
}

func TestNewEKFRejectsBadInitialCovariance(t *testing.T) {
	motion := &model.ConstantVelocity{QPSD: 0.05}
	cases := map[string]*model.State{
		"non-finite P": {Time: 0, X: mat.Vec(0, 0, 0, 0), P: mat.From([][]float64{{1, 0, 0, 0}, {0, math.NaN(), 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}})},
		"asymmetric P": {Time: 0, X: mat.Vec(0, 0, 0, 0), P: mat.From([][]float64{{1, 0.2, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}})},
		"indefinite P": {Time: 0, X: mat.Vec(0, 0, 0, 0), P: mat.From([][]float64{{1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}})},
		"non-finite x": {Time: 0, X: mat.Vec(math.Inf(1), 0, 0, 0), P: mat.Identity(4)},
	}
	for name, st := range cases {
		if _, err := NewEKF(st, motion, 1e-9, 1e12); err == nil {
			t.Fatalf("%s: expected rejection, got nil", name)
		}
	}
}

func TestPredictCovariancePSD(t *testing.T) {
	f, err := NewEKF(initialState(), &model.ConstantVelocity{QPSD: 0.05}, 1e-9, 1e12)
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []float64{0.1, 0.5, 1.0, 3.0, 10.0} {
		if err := f.Predict(tt); err != nil {
			t.Fatalf("predict %v: %v", tt, err)
		}
		assertPSD(t, f.State().P, "predicted P")
	}
	// zero process noise stays PSD
	f2, _ := NewEKF(initialState(), &model.ConstantVelocity{QPSD: 0}, 1e-9, 1e12)
	if err := f2.Predict(5); err != nil {
		t.Fatal(err)
	}
	assertPSD(t, f2.State().P, "P with Q=0")

	// backwards prediction rejected
	if err := f.Predict(1.0); err == nil {
		t.Fatal("backwards prediction must be rejected")
	}
}

func TestLinearUpdateShrinksPositionVariance(t *testing.T) {
	f, _ := NewEKF(initialState(), &model.ConstantVelocity{QPSD: 0.05}, 1e-9, 1e12)
	before := f.State().P.At(0, 0)
	pos := &model.LinearPosition{R: diag2(0.01, 0.01)}
	if _, err := f.Update(mat.Vec(0.1, -0.1), pos); err != nil {
		t.Fatal(err)
	}
	after := f.State().P.At(0, 0)
	if !(after < before) {
		t.Fatalf("position variance did not shrink: before=%.5f after=%.5f", before, after)
	}
	assertPSD(t, f.State().P, "post-update P")
}

func TestNonlinearUpdateAndReproducibility(t *testing.T) {
	run := func() []float64 {
		f, _ := NewEKF(initialState(), &model.ConstantVelocity{QPSD: 0.05}, 1e-9, 1e12)
		rb := &model.RangeBearing{Sx: -3, Sy: -2, R: diag2(0.04, 0.0004)}
		// expected measurement from true state near (0,0)
		_ = f.Predict(0.5)
		st := f.State()
		px, py := mat.VecAt(st.X, 0), mat.VecAt(st.X, 1)
		z := mat.Vec(math.Hypot(px+3, py+2), math.Atan2(py+2, px+3))
		if _, err := f.Update(z, rb); err != nil {
			t.Fatal(err)
		}
		s := f.State()
		return append(append([]float64{}, s.X.Data...), flatten(s.P)...)
	}
	a := run()
	b := run()
	if len(a) != len(b) {
		t.Fatal("length mismatch")
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-15 {
			t.Fatalf("nonlinear run not reproducible at %d: %.12f vs %.12f", i, a[i], b[i])
		}
	}
}

func TestUpdateRejectsIllegalNoise(t *testing.T) {
	motion := &model.ConstantVelocity{QPSD: 0.05}
	f, _ := NewEKF(initialState(), motion, 1e-9, 1e12)
	cases := map[string]model.MeasurementModel{
		"R not PSD":    &model.LinearPosition{R: mat.From([][]float64{{1, 0}, {0, -0.01}})},
		"R asymmetric": &model.LinearPosition{R: mat.From([][]float64{{1, 0.3}, {0.1, 1}})},
		"R non-finite": &model.LinearPosition{R: diag2(math.NaN(), 1)},
		"R nil":        &model.LinearPosition{R: nil},
	}
	for name, md := range cases {
		if _, err := f.Update(mat.Vec(0, 0), md); err == nil {
			t.Fatalf("%s: expected error", name)
		}
	}
	// state must be untouched after all those rejected updates
	st := f.State()
	if mat.VecAt(st.X, 0) != 0 {
		t.Fatal("rejected updates mutated state")
	}
}

func TestUpdateRejectsDegenerateAndIllConditioned(t *testing.T) {
	motion := &model.ConstantVelocity{QPSD: 0.05}
	// Truly degenerate: zero prior variance in x and R_xx=0 make S singular.
	st := initialState()
	st.P = mat.From([][]float64{
		{0, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 0.25, 0},
		{0, 0, 0, 0.25},
	})
	f, _ := NewEKF(st, motion, 1e-9, 1e12)
	md := &model.LinearPosition{R: diag2(0, 0.5)}
	if _, err := f.Update(mat.Vec(0, 0), md); err == nil {
		t.Fatal("singular innovation covariance must be rejected as degenerate")
	}

	// Conditioning limit: a wildly scaled R ratio.
	f2, _ := NewEKF(initialState(), motion, 1e-9, 1e6)
	md2 := &model.LinearPosition{R: diag2(1e12, 1e-6)}
	if _, err := f2.Update(mat.Vec(0, 0), md2); err == nil {
		t.Fatal("ill-conditioned S must be rejected")
	}
}

func TestNonFiniteMeasurementRejected(t *testing.T) {
	f, _ := NewEKF(initialState(), &model.ConstantVelocity{QPSD: 0.05}, 1e-9, 1e12)
	md := &model.LinearPosition{R: diag2(0.01, 0.01)}
	if _, err := f.Update(mat.Vec(math.NaN(), 0), md); err == nil {
		t.Fatal("NaN measurement must be rejected")
	}
	if _, err := f.Update(mat.Vec(0, math.Inf(1)), md); err == nil {
		t.Fatal("Inf measurement must be rejected")
	}
}

func TestRangeBearingSingularPoint(t *testing.T) {
	f, _ := NewEKF(initialState(), &model.ConstantVelocity{QPSD: 0.05}, 1e-9, 1e12)
	// sensor exactly at state position -> range zero, undefined bearing.
	rb := &model.RangeBearing{Sx: 0, Sy: 0, R: diag2(0.04, 0.0004)}
	if _, err := f.Update(mat.Vec(0, 0), rb); err == nil {
		t.Fatal("range-bearing at singular point must be rejected")
	}
}

func flatten(m *mat.Matrix) []float64 {
	out := make([]float64, len(m.Data))
	copy(out, m.Data)
	return out
}
