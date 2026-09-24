package model

import (
	"math"
	"testing"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
)

func TestConstantVelocityPredict(t *testing.T) {
	m := &ConstantVelocity{QPSD: 0.1}
	x := mat.Vec(1, 2, 0.5, -0.5)
	fx, F, Q, err := m.Predict(x, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	// expected new position: x + v*dt
	want := mat.Vec(1.25, 1.75, 0.5, -0.5)
	if !mat.Equal(fx, want, 1e-12) {
		t.Fatalf("fx=%v want %v", fx.Data, want.Data)
	}
	if F.At(0, 2) != 0.5 || F.At(1, 3) != 0.5 {
		t.Fatal("F structure wrong")
	}
	if !mat.IsSymmetric(Q, 1e-12) {
		t.Fatal("Q not symmetric")
	}
	if ok, _ := mat.IsPSD(Q, 1e-10); !ok {
		t.Fatal("Q not PSD")
	}
	// invalid inputs rejected
	for _, dt := range []float64{-1, math.NaN(), math.Inf(1)} {
		if _, _, _, err := m.Predict(x, dt); err == nil {
			t.Fatalf("dt=%v must be rejected", dt)
		}
	}
	bad := &ConstantVelocity{QPSD: -1}
	if _, _, _, err := bad.Predict(x, 0.1); err == nil {
		t.Fatal("negative QPSD must be rejected")
	}
}

func TestRangeBearingMeasure(t *testing.T) {
	rb := &RangeBearing{Sx: 0, Sy: 0, R: mat.From([][]float64{{0.01, 0}, {0, 0.01}})}
	x := mat.Vec(3, 4, 0, 0)
	hx, H, err := rb.Measure(x)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(mat.VecAt(hx, 0)-5) > 1e-12 {
		t.Fatalf("range=%v want 5", mat.VecAt(hx, 0))
	}
	if math.Abs(mat.VecAt(hx, 1)-math.Atan2(4, 3)) > 1e-12 {
		t.Fatal("bearing wrong")
	}
	// Jacobian: dr/dx = dx/r = 3/5
	if math.Abs(H.At(0, 0)-0.6) > 1e-12 {
		t.Fatalf("H range row wrong: %v", H.Slice(0))
	}
	// singular point
	zero := mat.Vec(0, 0, 0, 0)
	if _, _, err := rb.Measure(zero); err == nil {
		t.Fatal("singular point must error")
	}
}

func TestLinearPositionMeasure(t *testing.T) {
	lp := &LinearPosition{R: mat.From([][]float64{{0.1, 0}, {0, 0.1}})}
	hx, H, err := lp.Measure(mat.Vec(7, 8, 1, -1))
	if err != nil {
		t.Fatal(err)
	}
	if !mat.Equal(hx, mat.Vec(7, 8), 1e-12) {
		t.Fatalf("hx=%v", hx.Data)
	}
	if H.Rows != 2 || H.Cols != 4 {
		t.Fatal("H shape wrong")
	}
}
