package engine

import (
	"errors"
	"fmt"
	"math"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/model"
)

// EKF is a single-threaded extended Kalman filter.
type EKF struct {
	state     *model.State
	motion    model.MotionModel
	eigTol    float64
	condLimit float64
}

// NewEKF constructs a filter from an initial state. The initial covariance
// must be finite, symmetric and positive semidefinite; the mean must be
// finite. eigTol is the relative PSD tolerance and condLimit the maximum
// acceptable innovation-covariance condition number.
func NewEKF(initial *model.State, motion model.MotionModel, eigTol, condLimit float64) (*EKF, error) {
	if initial == nil || initial.X == nil || initial.P == nil {
		return nil, errors.New("ekf: nil initial state")
	}
	if motion == nil {
		return nil, errors.New("ekf: nil motion model")
	}
	n := motion.Dim()
	if initial.X.Rows != n || initial.P.Rows != n || initial.P.Cols != n {
		return nil, fmt.Errorf("%w: initial state/model dimension", ErrDimensionMismatch)
	}
	if !mat.Finite(initial.X) {
		return nil, fmt.Errorf("%w: initial mean contains NaN/Inf", ErrNonFinite)
	}
	if !mat.Finite(initial.P) {
		return nil, fmt.Errorf("%w: initial covariance contains NaN/Inf", ErrNonFinite)
	}
	if !mat.IsSymmetric(initial.P, eigTol) {
		return nil, fmt.Errorf("%w: initial covariance not symmetric", ErrIllegalNoise)
	}
	if ok, err := mat.IsPSD(initial.P, eigTol); err != nil || !ok {
		return nil, fmt.Errorf("%w: initial covariance not PSD", ErrIllegalNoise)
	}
	if math.IsNaN(initial.Time) || math.IsInf(initial.Time, 0) {
		return nil, errors.New("ekf: initial time not finite")
	}
	return &EKF{
		state:     initial.Clone(),
		motion:    motion,
		eigTol:    eigTol,
		condLimit: condLimit,
	}, nil
}

// Time returns the current state timestamp.
func (f *EKF) Time() float64 { return f.state.Time }

// State returns a copy of the current state.
func (f *EKF) State() *model.State { return f.state.Clone() }

// Predict propagates the state to time t. t must be finite and >= current
// time; dt == 0 is a no-op.
func (f *EKF) Predict(t float64) error {
	if math.IsNaN(t) || math.IsInf(t, 0) {
		return errors.New("ekf: target time not finite")
	}
	dt := t - f.state.Time
	if dt < -f.eigTol {
		return fmt.Errorf("ekf: predict backwards in time %.6f -> %.6f", f.state.Time, t)
	}
	if dt <= 0 {
		return nil
	}
	fx, F, Q, err := f.motion.Predict(f.state.X, dt)
	if err != nil {
		return fmt.Errorf("ekf: motion model: %w", err)
	}
	if F.Rows != f.motion.Dim() || F.Cols != f.motion.Dim() ||
		Q.Rows != f.motion.Dim() || Q.Cols != f.motion.Dim() {
		return fmt.Errorf("%w: F/Q dimensions", ErrDimensionMismatch)
	}
	if !mat.Finite(F) || !mat.Finite(Q) || !mat.Finite(fx) {
		return fmt.Errorf("%w: non-finite prediction terms", ErrNonFinite)
	}
	if !mat.IsSymmetric(Q, f.eigTol) {
		return fmt.Errorf("%w: process noise Q not symmetric", ErrIllegalNoise)
	}
	if ok, _ := mat.IsPSD(Q, f.eigTol); !ok {
		return fmt.Errorf("%w: process noise Q not PSD", ErrIllegalNoise)
	}
	n := f.motion.Dim()
	// P = F P F^T + Q, then symmetrize to kill round-off asymmetry.
	FP := mat.Mul(F, f.state.P)
	FPFt := mat.Mul(FP, mat.T(F))
	P := mat.Add(FPFt, Q)
	P = mat.Symmetrize(P)
	if !mat.Finite(P) {
		return fmt.Errorf("%w: predicted covariance diverged", ErrNonFinite)
	}
	f.state.X = fx
	f.state.P = P
	f.state.Time = t
	_ = n
	return nil
}

// Update performs one measurement update using either a linear or a
// nonlinear measurement model. It returns the normalized innovation
// squared (NIS = y^T S^{-1} y) for fault detection.
func (f *EKF) Update(z *mat.Vector, md model.MeasurementModel) (float64, error) {
	b, err := f.prepareInnovation(z, md)
	if err != nil {
		return 0, err
	}
	n := f.motion.Dim()
	m := md.Dim()
	R := md.Noise()
	_, H, err := md.Measure(f.state.X)
	if err != nil {
		return 0, fmt.Errorf("%w: %v", ErrBadModel, err)
	}
	Sinv, err := mat.InvertSPD(b.S)
	if err != nil {
		return 0, fmt.Errorf("%w: S not invertible: %v", ErrDegenerate, err)
	}

	// K = P H^T S^{-1}.
	K := mat.Mul(mat.Mul(f.state.P, mat.T(H)), Sinv)

	// x = x + K y.
	newX := mat.Add(f.state.X, mat.MulVec(K, b.y))

	// Joseph form covariance: P = (I-KH) P (I-KH)^T + K R K^T, guaranteed
	// symmetric PSD as long as R is PSD.
	I := mat.Identity(n)
	KH := mat.Mul(K, H)
	A := mat.Sub(I, KH)
	APA := mat.Mul(mat.Mul(A, f.state.P), mat.T(A))
	KRK := mat.Mul(mat.Mul(K, R), mat.T(K))
	newP := mat.Symmetrize(mat.Add(APA, KRK))
	if !mat.Finite(newX) || !mat.Finite(newP) {
		return 0, fmt.Errorf("%w: post-update state non-finite", ErrNonFinite)
	}
	if ok, _ := mat.IsPSD(newP, f.eigTol); !ok {
		return 0, fmt.Errorf("%w: post-update covariance lost PSD", ErrIllegalNoise)
	}

	_ = m
	f.state.X = newX
	f.state.P = newP
	return b.nis, nil
}

func dot(a, b *mat.Vector) float64 {
	var s float64
	for i := range a.Data {
		s += a.Data[i] * b.Data[i]
	}
	return s
}

func dimOr0(m *mat.Matrix) int {
	if m == nil {
		return 0
	}
	return m.Rows
}

func dimOr0c(m *mat.Matrix) int {
	if m == nil {
		return 0
	}
	return m.Cols
}

// innovationBundle holds the quantities needed by both the mutating update
// and the read-only NIS probe.
type innovationBundle struct {
	y   *mat.Vector
	S   *mat.Matrix
	nis float64
}

// prepareInnovation validates inputs, evaluates h(x)/H, builds S and computes
// NIS without changing the filter state. It is shared by Update and NIS.
func (f *EKF) prepareInnovation(z *mat.Vector, md model.MeasurementModel) (*innovationBundle, error) {
	n := f.motion.Dim()
	m := md.Dim()
	if z == nil {
		return nil, fmt.Errorf("%w: nil measurement", ErrDimensionMismatch)
	}
	if z.Rows != m || z.Cols != 1 {
		return nil, fmt.Errorf("%w: measurement is %dx1, model dim %d", ErrDimensionMismatch, z.Rows, m)
	}
	if !mat.Finite(z) {
		return nil, fmt.Errorf("%w: measurement contains NaN/Inf", ErrNonFinite)
	}
	R := md.Noise()
	if R == nil || R.Rows != m || R.Cols != m {
		return nil, fmt.Errorf("%w: R dimension", ErrDimensionMismatch)
	}
	if !mat.Finite(R) {
		return nil, fmt.Errorf("%w: R contains NaN/Inf", ErrIllegalNoise)
	}
	if !mat.IsSymmetric(R, f.eigTol) {
		return nil, fmt.Errorf("%w: R not symmetric", ErrIllegalNoise)
	}
	if ok, _ := mat.IsPSD(R, f.eigTol); !ok {
		return nil, fmt.Errorf("%w: R not positive semidefinite", ErrIllegalNoise)
	}

	hx, H, err := md.Measure(f.state.X)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrBadModel, err)
	}
	if hx == nil || H == nil || H.Rows != m || H.Cols != n || hx.Rows != m {
		return nil, fmt.Errorf("%w: h/H dimensions", ErrDimensionMismatch)
	}
	if !mat.Finite(hx) || !mat.Finite(H) {
		return nil, fmt.Errorf("%w: non-finite h(x) or H", ErrBadModel)
	}

	y := mat.Sub(z, hx)
	if !mat.Finite(y) {
		return nil, fmt.Errorf("%w: non-finite innovation", ErrNonFinite)
	}
	S := mat.Symmetrize(mat.Add(mat.Mul(mat.Mul(H, f.state.P), mat.T(H)), R))
	if !mat.Finite(S) {
		return nil, fmt.Errorf("%w: S non-finite", ErrNonFinite)
	}
	if ok, _ := mat.IsPSD(S, f.eigTol); !ok {
		return nil, fmt.Errorf("%w: S not positive semidefinite", ErrIllegalNoise)
	}
	eig, err := mat.EigenSymmetric(S)
	if err != nil {
		return nil, fmt.Errorf("%w: cannot inspect S", ErrIllConditioned)
	}
	lMin, lMax := eig[0], eig[len(eig)-1]
	if lMin <= 0 {
		return nil, fmt.Errorf("%w: S has non-positive eigenvalue %.3e", ErrDegenerate, lMin)
	}
	cond := lMax / lMin
	if cond > f.condLimit || math.IsInf(cond, 0) || math.IsNaN(cond) {
		return nil, fmt.Errorf("%w: cond(S)=%.3e > limit %.3e", ErrIllConditioned, cond, f.condLimit)
	}
	Sinv, err := mat.InvertSPD(S)
	if err != nil {
		return nil, fmt.Errorf("%w: S not invertible: %v", ErrDegenerate, err)
	}
	nis := dot(y, mat.MulVec(Sinv, y))
	if nis < 0 {
		nis = 0
	}
	return &innovationBundle{y: y, S: S, nis: nis}, nil
}

// NIS evaluates the normalized innovation squared for a measurement against
// the current state WITHOUT updating it. Used for recovery probes on isolated
// streams. The boolean is false on any structural/numerical failure.
func (f *EKF) NIS(z *mat.Vector, md model.MeasurementModel) (float64, bool) {
	b, err := f.prepareInnovation(z, md)
	if err != nil || b == nil {
		return 0, false
	}
	return b.nis, true
}
