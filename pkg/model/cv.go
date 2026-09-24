package model

import (
	"errors"
	"math"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
)

// ConstantVelocity is a 2D constant-velocity motion model with state
// [px, py, vx, vy]. Process noise is continuous white-noise acceleration with
// power spectral density QPSD:
//
//	Q = QPSD * [ dt^3/3 I2  dt^2/2 I2
//	            dt^2/2 I2  dt     I2 ]
type ConstantVelocity struct {
	// QPSD is the acceleration noise power spectral density (>= 0).
	QPSD float64
}

// Dim returns 4.
func (m *ConstantVelocity) Dim() int { return 4 }

// Predict implements MotionModel.
func (m *ConstantVelocity) Predict(x *mat.Vector, dt float64) (*mat.Matrix, *mat.Matrix, *mat.Matrix, error) {
	if x == nil || x.Rows != 4 {
		return nil, nil, nil, errors.New("cv: state must be a 4-vector")
	}
	if math.IsNaN(dt) || math.IsInf(dt, 0) || dt < 0 {
		return nil, nil, nil, errors.New("cv: dt must be a finite non-negative number")
	}
	if math.IsNaN(m.QPSD) || math.IsInf(m.QPSD, 0) || m.QPSD < 0 {
		return nil, nil, nil, errors.New("cv: QPSD must be finite and non-negative")
	}

	F := mat.From([][]float64{
		{1, 0, dt, 0},
		{0, 1, 0, dt},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
	})
	fx := mat.MulVec(F, x)

	q := m.QPSD
	d := dt
	Q := mat.From([][]float64{
		{q * d * d * d / 3, 0, q * d * d / 2, 0},
		{0, q * d * d * d / 3, 0, q * d * d / 2},
		{q * d * d / 2, 0, q * d, 0},
		{0, q * d * d / 2, 0, q * d},
	})
	return fx, F, Q, nil
}

var _ MotionModel = (*ConstantVelocity)(nil)
