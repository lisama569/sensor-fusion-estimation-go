// Package models defines the state transition model and the linear /
// nonlinear measurement models used by the estimator.
package models

import (
	"math"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/noise"
	"github.com/lisama569/sensor-fusion-estimation-go/internal/reject"
	"gonum.org/v1/gonum/mat"
)

// StateModel is a linear time-invariant continuous-discrete base model
//
//	x_{k+1} = F(dt) x_k + w,  w ~ N(0, Q(dt)).
//
// F0 is the per-unit-time transition companion and Q0 is the process noise
// accumulated over one unit of time; the filter scales them to arbitrary dt.
type StateModel struct {
	F0 *mat.Dense
	Q0 *mat.SymDense
	n  int
}

// Nx returns the state dimension.
func (m *StateModel) Nx() int { return m.n }

// TransitionBase returns the per-unit-time F.
func (m *StateModel) TransitionBase() *mat.Dense { return m.F0 }

// ProcessNoiseBase returns the per-unit-time Q.
func (m *StateModel) ProcessNoiseBase() *mat.SymDense { return m.Q0 }

// NewStateModel validates F and Q and builds the model. Q must be finite
// symmetric positive semidefinite (zero process noise is legal).
func NewStateModel(F *mat.Dense, Q *mat.SymDense) (*StateModel, error) {
	fr, fc := F.Dims()
	if fr != fc {
		return nil, reject.New(reject.ReasonDimensionMismatch, "F must be square, got %dx%d", fr, fc)
	}
	qr, qc := Q.Dims()
	if qr != qc || qr != fr {
		return nil, reject.New(reject.ReasonDimensionMismatch, "Q must be %dx%d, got %dx%d", fr, fr, qr, qc)
	}
	cfg := noise.DefaultConfig()
	// F is a (generally non-symmetric) transition matrix; only finiteness
	// applies. Q must be symmetric PSD.
	if err := noise.ValidateFinite(F); err != nil {
		return nil, err
	}
	if err := noise.ValidateSPD(Q, cfg); err != nil {
		return nil, err
	}
	return &StateModel{F0: mat.DenseCopyOf(F), Q0: copySym(Q), n: fr}, nil
}

// NewConstantVelocity2D builds a 4D model with state [x, vx, y, vy].
//
//	F0 (per unit dt): [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1] at dt=1,
//	Q0 is the discrete white-noise-acceleration covariance for spectral
//	density q over one unit of time; Q(dt) is produced by the filter.
func NewConstantVelocity2D(q float64) *StateModel {
	if math.IsNaN(q) || math.IsInf(q, 0) || q < 0 {
		panic(reject.New(reject.ReasonNoiseNotSPD, "q must be a finite non-negative number, got %v", q))
	}
	F := mat.NewDense(4, 4, []float64{
		1, 1, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 1,
		0, 0, 0, 1,
	})
	// Q(dt) for white-noise acceleration with spectral density q:
	// per axis block: q*[dt^4/4 dt^3/2; dt^3/2 dt^2], evaluated at dt=1.
	// NewSymDense reads the upper triangle only.
	d := []float64{
		q / 4, q / 2, 0, 0,
		0, q, 0, 0,
		0, 0, q / 4, q / 2,
		0, 0, 0, q,
	}
	Q := mat.NewSymDense(4, d)
	m, err := NewStateModel(F, Q)
	if err != nil {
		panic(err)
	}
	return m
}

func copySym(s *mat.SymDense) *mat.SymDense {
	n, _ := s.Dims()
	out := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			out.SetSym(i, j, s.At(i, j))
		}
	}
	return out
}

// Measurement is one timestamped observation produced by a sensor channel.
type Measurement struct {
	SensorID  string
	Timestamp float64
	Z         *mat.VecDense
}

// MeasureModel maps a state to an expected measurement and provides the
// observation matrix (constant H for linear models, Jacobian otherwise).
type MeasureModel interface {
	// M is the measurement dimension.
	M() int
	// H returns the observation matrix / Jacobian at state x.
	H(x *mat.VecDense) *mat.Dense
	// Observe returns h(x) (without the constant offset).
	Observe(x *mat.VecDense) mat.Vector
	// Offset returns the constant b in z = h(x) + b + v (nil means zero).
	Offset() mat.Vector
	// R returns the measurement noise covariance.
	R() *mat.SymDense
}

func validateMeasureParts(mDim, nDim int, H *mat.Dense, b mat.Vector, R *mat.SymDense) error {
	hr, hc := H.Dims()
	if hr != mDim || hc != nDim {
		return reject.New(reject.ReasonDimensionMismatch,
			"H must be %dx%d, got %dx%d", mDim, nDim, hr, hc)
	}
	if b != nil && b.Len() != mDim {
		return reject.New(reject.ReasonDimensionMismatch,
			"offset length must be %d, got %d", mDim, b.Len())
	}
	if r, c := R.Dims(); r != mDim || c != mDim {
		return reject.New(reject.ReasonDimensionMismatch,
			"R must be %dx%d, got %dx%d", mDim, mDim, r, c)
	}
	cfg := noise.DefaultConfig()
	if err := noise.ValidateFinite(H); err != nil {
		return err
	}
	if err := noise.ValidateSPD(R, cfg); err != nil {
		return err
	}
	return nil
}

// AngleWrapper is an optional capability implemented by measurement models
// whose vector contains angular components. Rows listed by AngleRows are
// wrapped into [-pi, pi) after subtracting the predicted measurement, which
// makes range-bearing updates correct across the +/-pi cut.
type AngleWrapper interface {
	AngleRows() []int
}

// LinearModel is z = H x + b + v, v ~ N(0, R).
type LinearModel struct {
	Hm *mat.Dense
	B  mat.Vector
	Rm *mat.SymDense
	m  int
}

func (l *LinearModel) M() int                       { return l.m }
func (l *LinearModel) H(_ *mat.VecDense) *mat.Dense { return l.Hm }
func (l *LinearModel) Offset() mat.Vector           { return l.B }

func (l *LinearModel) Observe(x *mat.VecDense) mat.Vector {
	out := mat.NewVecDense(l.m, nil)
	out.MulVec(l.Hm, x)
	return out
}

func (l *LinearModel) R() *mat.SymDense { return l.Rm }

// NewLinearModel validates and builds a linear measurement model. nDim is
// the state dimension. R must be finite symmetric positive semidefinite
// (a singular R is legal; update-time degeneracy is reported separately).
func NewLinearModel(nDim int, H *mat.Dense, offset mat.Vector, R *mat.SymDense) (*LinearModel, error) {
	if H == nil || R == nil {
		return nil, reject.New(reject.ReasonDimensionMismatch, "H and R must be non-nil")
	}
	mDim, _ := H.Dims()
	if err := validateMeasureParts(mDim, nDim, H, offset, R); err != nil {
		return nil, err
	}
	lm := &LinearModel{
		Hm: mat.DenseCopyOf(H),
		Rm: copySym(R),
		m:  mDim,
	}
	if offset != nil {
		lm.B = mat.VecDenseCopyOf(offset)
	}
	return lm, nil
}

// NonlinearModel is z = h(x) + b + v with a state-dependent Jacobian.
type NonlinearModel struct {
	Mdim     int
	Eval     func(x *mat.VecDense) *mat.VecDense
	Jacobian func(x *mat.VecDense) *mat.Dense
	B        mat.Vector
	Rm       *mat.SymDense
	Angles   []int
}

// AngleRows implements AngleWrapper.
func (nl *NonlinearModel) AngleRows() []int { return nl.Angles }

func (nl *NonlinearModel) M() int                       { return nl.Mdim }
func (nl *NonlinearModel) H(x *mat.VecDense) *mat.Dense { return nl.Jacobian(x) }
func (nl *NonlinearModel) Offset() mat.Vector           { return nl.B }
func (nl *NonlinearModel) R() *mat.SymDense             { return nl.Rm }

func (nl *NonlinearModel) Observe(x *mat.VecDense) mat.Vector { return nl.Eval(x) }

// NewNonlinearModel validates and builds a nonlinear measurement model.
func NewNonlinearModel(nDim, mDim int, eval func(*mat.VecDense) *mat.VecDense,
	jac func(*mat.VecDense) *mat.Dense, offset mat.Vector, R *mat.SymDense,
	angleRows ...int) (*NonlinearModel, error) {
	var AnglesFromArgs []int = angleRows
	if eval == nil || jac == nil || R == nil {
		return nil, reject.New(reject.ReasonDimensionMismatch, "eval, jacobian and R must be non-nil")
	}
	probe := mat.NewVecDense(nDim, nil)
	H := jac(probe)
	hx := eval(probe)
	if H == nil || hx == nil {
		return nil, reject.New(reject.ReasonDimensionMismatch, "jacobian/eval returned nil")
	}
	if err := validateMeasureParts(mDim, nDim, H, offset, R); err != nil {
		return nil, err
	}
	if hx.Len() != mDim {
		return nil, reject.New(reject.ReasonDimensionMismatch,
			"eval output length must be %d, got %d", mDim, hx.Len())
	}
	nl := &NonlinearModel{
		Mdim:     mDim,
		Eval:     eval,
		Jacobian: jac,
		Rm:       copySym(R),
	}
	if offset != nil {
		nl.B = mat.VecDenseCopyOf(offset)
	}
	nl.Angles = append([]int(nil), AnglesFromArgs...)
	return nl, nil
}
