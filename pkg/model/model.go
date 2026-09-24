// Package model defines the state-space models used by the estimation engine:
// a motion (process) model and linear or nonlinear measurement models.
package model

import (
	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
)

// State is an estimated state at a point in time.
type State struct {
	// Time is the state timestamp in arbitrary monotonic seconds.
	Time float64
	// X is the state mean column vector (n x 1).
	X *mat.Vector
	// P is the state covariance matrix (n x n), symmetric PSD.
	P *mat.Matrix
}

// Clone returns an independent copy of the state.
func (s *State) Clone() *State {
	return &State{Time: s.Time, X: s.X.Clone(), P: s.P.Clone()}
}

// MotionModel propagates state and covariance over a time step.
type MotionModel interface {
	// Dim is the state dimension n.
	Dim() int
	// Predict returns F*x and the state transition Jacobian F, the process
	// noise covariance Q (symmetric PSD, n x n), for a step of dt seconds.
	Predict(x *mat.Vector, dt float64) (fx *mat.Vector, F, Q *mat.Matrix, err error)
}

// MeasurementModel maps state to an expected measurement. Linear models
// return a constant Jacobian; nonlinear models evaluate it at the state.
type MeasurementModel interface {
	// Dim is the measurement dimension m.
	Dim() int
	// Measure returns h(x) and the measurement Jacobian H (m x n).
	Measure(x *mat.Vector) (hx *mat.Vector, H *mat.Matrix, err error)
	// Noise returns the measurement noise covariance R (m x m),
	// symmetric PSD.
	Noise() *mat.Matrix
}
