package models

import (
	"math"
	"testing"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/reject"
	"gonum.org/v1/gonum/mat"
)

func TestNewStateModelRejectsBadShapesAndNoise(t *testing.T) {
	goodF := mat.NewDense(2, 2, []float64{1, 1, 0, 1})
	goodQ := mat.NewSymDense(2, []float64{0.1, 0, 0, 0.1})

	// F non-square.
	_, err := NewStateModel(mat.NewDense(2, 3, nil), goodQ)
	if reject.Of(err) != reject.ReasonDimensionMismatch {
		t.Fatalf("want dimension_mismatch for non-square F, got %v", err)
	}
	t.Logf("decision: 2x3 F rejected: %v", err)

	// Q wrong size.
	badQ := mat.NewSymDense(3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1})
	_, err = NewStateModel(goodF, badQ)
	if reject.Of(err) != reject.ReasonDimensionMismatch {
		t.Fatalf("want dimension_mismatch for Q size, got %v", err)
	}
	t.Logf("decision: 3x3 Q for 2D model rejected: %v", err)

	// Q indefinite.
	indefQ := mat.NewSymDense(2, []float64{1, 0, 0, -1})
	_, err = NewStateModel(goodF, indefQ)
	if reject.Of(err) != reject.ReasonNoiseNotSPD {
		t.Fatalf("want noise_not_spd for indefinite Q, got %v", err)
	}
	t.Logf("decision: indefinite Q rejected: %v", err)

	// Q with NaN.
	nanQ := mat.NewSymDense(2, []float64{1, 0, 0, math.NaN()})
	_, err = NewStateModel(goodF, nanQ)
	if reject.Of(err) != reject.ReasonNoiseNotSPD {
		t.Fatalf("want noise_not_spd for NaN Q, got %v", err)
	}
	t.Logf("decision: NaN Q rejected: %v", err)

	// Semidefinite Q (zero velocity noise) is legal.
	if _, err := NewStateModel(goodF, mat.NewSymDense(2, []float64{0.1, 0, 0, 0})); err != nil {
		t.Fatalf("semidefinite Q should be legal: %v", err)
	}
	t.Log("decision: semidefinite Q diag(0.1,0) accepted")
}

func TestNewLinearModelRejectsMismatchAndNoise(t *testing.T) {
	goodH := mat.NewDense(1, 2, []float64{1, 0})
	goodR := mat.NewSymDense(1, []float64{0.5})

	// H width != state dimension.
	_, err := NewLinearModel(3, goodH, nil, goodR)
	if reject.Of(err) != reject.ReasonDimensionMismatch {
		t.Fatalf("want dimension_mismatch for H width, got %v", err)
	}
	t.Logf("decision: 1x2 H for 3D state rejected: %v", err)

	// R indefinite.
	_, err = NewLinearModel(2, goodH, nil, mat.NewSymDense(1, []float64{-1}))
	if reject.Of(err) != reject.ReasonNoiseNotSPD {
		t.Fatalf("want noise_not_spd for negative R, got %v", err)
	}
	t.Logf("decision: R=-1 rejected: %v", err)

	// offset length mismatch.
	_, err = NewLinearModel(2, goodH, mat.NewVecDense(2, []float64{0, 0}), goodR)
	if reject.Of(err) != reject.ReasonDimensionMismatch {
		t.Fatalf("want dimension_mismatch for offset length, got %v", err)
	}
	t.Logf("decision: 2-vector offset for 1D measurement rejected: %v", err)
}

func TestConstantVelocity2D(t *testing.T) {
	m := NewConstantVelocity2D(0.5)
	if m.Nx() != 4 {
		t.Fatalf("want 4D, got %d", m.Nx())
	}
	t.Log("decision: constant velocity 2D model builds with 4D state [x,vx,y,vy]")

	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("negative q must panic")
			} else {
				t.Logf("decision: negative q rejected at construction: %v", r)
			}
		}()
		_ = NewConstantVelocity2D(-1)
	}()
}
