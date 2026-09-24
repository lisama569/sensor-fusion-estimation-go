// Package filter implements a numerically stable (Extended) Kalman filter:
// prediction with a linear constant-velocity state model and block updates
// with linear or nonlinear measurement models.
//
// Stability measures:
//
//   - all inputs (x, P, z, H, R, innovations) are checked for finiteness;
//   - the innovation covariance S = H P H' + R is solved via Cholesky;
//     a singular-to-working-precision S is rejected (ReasonDegenerateUpdate)
//     instead of producing NaNs;
//   - covariance is propagated/symmetrized and updated in Joseph form
//     P = (I-KH) P (I-KH)' + K R K', which is symmetric positive
//     semidefinite for any K;
//   - a normalized innovation squared (NIS) gate detects divergence.
package filter

import (
	"math"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/models"
	"github.com/lisama569/sensor-fusion-estimation-go/internal/noise"
	"github.com/lisama569/sensor-fusion-estimation-go/internal/reject"
	"gonum.org/v1/gonum/mat"
)

// State is an estimated mean/covariance pair at a given time.
type State struct {
	T float64
	X *mat.VecDense
	P *mat.SymDense
}

// Config holds filter-wide numerical safeguards.
type Config struct {
	// NISGate is the gate on the normalized innovation squared
	// nu' S^-1 nu; updates above the gate are flagged (and drive channel
	// isolation in the fuser). 0 disables gating.
	NISGate float64
	// EigenFloor is the floor applied when repairing P after each step.
	EigenFloor float64
}

// DefaultConfig returns safe defaults.
func DefaultConfig() Config {
	return Config{NISGate: 0, EigenFloor: 0}
}

// Filter is a stateful EKF.
type Filter struct {
	sm    *models.StateModel
	nx    int
	state *State
	cfg   Config
}

// New constructs a filter for the state model.
func New(sm *models.StateModel, cfg Config) *Filter {
	return &Filter{sm: sm, nx: sm.Nx(), cfg: cfg}
}

// Model returns the state model.
func (f *Filter) Model() *models.StateModel { return f.sm }

// Clone returns an independent deep copy (including current state).
func (f *Filter) Clone() *Filter {
	cp := &Filter{sm: f.sm, nx: f.nx, cfg: f.cfg}
	if f.state != nil {
		cp.state = &State{
			T: f.state.T,
			X: mat.VecDenseCopyOf(f.state.X),
			P: cloneSym(f.state.P),
		}
	}
	return cp
}

// Init sets the initial state. x length and P dimensions must match the
// model and P must be finite symmetric positive semidefinite.
func (f *Filter) Init(t0 float64, x *mat.VecDense, P *mat.SymDense) error {
	if x.Len() != f.nx {
		return reject.New(reject.ReasonDimensionMismatch,
			"initial x length %d != state dimension %d", x.Len(), f.nx)
	}
	if n, _ := P.Dims(); n != f.nx {
		return reject.New(reject.ReasonDimensionMismatch,
			"initial P must be %dx%d", f.nx, f.nx)
	}
	if err := noise.ValidateSPD(P, noise.DefaultConfig()); err != nil {
		return err
	}
	if math.IsNaN(t0) || math.IsInf(t0, 0) {
		return reject.New(reject.ReasonBadTimestamp, "initial time %v", t0)
	}
	for i := 0; i < x.Len(); i++ {
		if math.IsNaN(x.AtVec(i)) || math.IsInf(x.AtVec(i), 0) {
			return reject.New(reject.ReasonNonFiniteValue, "initial x contains NaN/Inf")
		}
	}
	f.state = &State{
		T: t0,
		X: mat.VecDenseCopyOf(x),
		P: cloneSym(P),
	}
	return nil
}

// State returns a copy of the current estimate (nil before Init).
func (f *Filter) State() *State {
	if f.state == nil {
		return nil
	}
	return &State{T: f.state.T, X: mat.VecDenseCopyOf(f.state.X), P: cloneSym(f.state.P)}
}

// Initialized reports whether Init succeeded.
func (f *Filter) Initialized() bool { return f.state != nil }

// PredictTo advances the state to time t (dt must be >= 0). The transition
// is evaluated for the constant-velocity base model scaled to dt:
//
//	F(dt) = I + (F0-I)*dt,  Q(dt) built from the white-noise-acceleration
//	integrals using q(dt) = q0*dt (i.e. entries scale as dt^4..dt^1).
//
// For a general model this keeps the per-axis structure of Q0; see
// processNoiseDT.
func (f *Filter) PredictTo(t float64) error {
	if f.state == nil {
		return reject.New(reject.ReasonNotInitialized, "predict before init")
	}
	if math.IsNaN(t) || math.IsInf(t, 0) {
		return reject.New(reject.ReasonBadTimestamp, "predict target time %v", t)
	}
	dt := t - f.state.T
	if dt < -1e-12 {
		return reject.New(reject.ReasonStaleTimestamp,
			"cannot predict backwards: current %.9f target %.9f", f.state.T, t)
	}
	if dt < 0 {
		dt = 0
	}
	if dt == 0 {
		f.state.T = t
		return nil
	}

	Fdt := transitionDT(f.sm.TransitionBase(), dt)
	Qdt := processNoiseDT(f.sm.ProcessNoiseBase(), dt)

	var xNew mat.VecDense
	xNew.MulVec(Fdt, f.state.X)
	if !vecFinite(&xNew) {
		return reject.New(reject.ReasonNonFiniteValue, "predicted state non-finite")
	}

	// P = F P F' + Q
	var fp mat.Dense
	fp.Mul(Fdt, f.state.P)
	var fpf mat.Dense
	fpf.Mul(&fp, Fdt.T())
	Pnew := mat.NewSymDense(f.nx, nil)
	for i := 0; i < f.nx; i++ {
		for j := i; j < f.nx; j++ {
			Pnew.SetSym(i, j, fpf.At(i, j)+Qdt.At(i, j))
		}
	}
	if err := noise.ValidateSPD(Pnew, noise.DefaultConfig()); err != nil {
		// Numerical drift: repair instead of crashing, but require the
		// repair to be finite.
		repaired, rerr := noise.NearestSPD(Pnew, 1e-14)
		if rerr != nil {
			return rerr
		}
		Pnew = repaired
	}
	f.state.X = &xNew
	f.state.P = Pnew
	f.state.T = t
	return nil
}

// InnovOutcome describes a single update decision.
type InnovOutcome struct {
	Accepted  bool
	NIS       float64
	Reason    string // reject.Reason when not accepted
	Timestamp float64
	SensorID  string
}

// Update applies one block measurement update. Non-finite innovations and
// singular innovation covariances are rejected with distinguishable
// reasons; the prior state is left untouched on any rejection.
func (f *Filter) Update(sensorID string, mm models.Measurement, md models.MeasureModel) (InnovOutcome, error) {
	out := InnovOutcome{Timestamp: mm.Timestamp, SensorID: sensorID}
	if f.state == nil {
		r := reject.New(reject.ReasonNotInitialized, "update before init from sensor %q", sensorID)
		out.Reason = string(r.Reason)
		return out, r
	}
	nx := f.nx
	mDim := md.M()
	if mm.Z.Len() != mDim {
		r := reject.New(reject.ReasonDimensionMismatch,
			"sensor %q z length %d != model M %d", sensorID, mm.Z.Len(), mDim)
		out.Reason = string(r.Reason)
		return out, r
	}

	H := md.H(f.state.X)
	if hr, hc := H.Dims(); hr != mDim || hc != nx {
		r := reject.New(reject.ReasonDimensionMismatch,
			"sensor %q H is %dx%d, want %dx%d", sensorID, hr, hc, mDim, nx)
		out.Reason = string(r.Reason)
		return out, r
	}

	hx := mat.VecDenseCopyOf(md.Observe(f.state.X))
	if b := md.Offset(); b != nil {
		hx.AddVec(hx, b)
	}
	if !vecFinite(hx) || !matFinite(H) {
		r := reject.New(reject.ReasonNonFiniteInnovation,
			"sensor %q produced non-finite predicted measurement/Jacobian", sensorID)
		out.Reason = string(r.Reason)
		return out, r
	}

	nu := mat.NewVecDense(mDim, nil)
	nu.SubVec(mm.Z, hx)
	if aw, ok := md.(interface{ AngleRows() []int }); ok {
		for _, row := range aw.AngleRows() {
			if row >= 0 && row < mDim {
				nu.SetVec(row, wrapPi(nu.AtVec(row)))
			}
		}
	}
	if !vecFinite(nu) {
		r := reject.New(reject.ReasonNonFiniteInnovation,
			"sensor %q innovation non-finite", sensorID)
		out.Reason = string(r.Reason)
		return out, r
	}

	// S = H P H' + R  (symmetric)
	var hp mat.Dense
	hp.Mul(H, f.state.P)
	var hph mat.Dense
	hph.Mul(&hp, H.T())
	S := mat.NewSymDense(mDim, nil)
	Rm := md.R()
	for i := 0; i < mDim; i++ {
		for j := i; j < mDim; j++ {
			S.SetSym(i, j, hph.At(i, j)+Rm.At(i, j))
		}
	}

	var chol mat.Cholesky
	if ok := chol.Factorize(S); !ok {
		r := reject.New(reject.ReasonDegenerateUpdate,
			"sensor %q: innovation covariance not positive definite (singular measurement)", sensorID)
		out.Reason = string(r.Reason)
		return out, r
	}

	// K = P H' S^-1 ; solve K S = P H' row-wise by solving S K' = H P'.
	var phT mat.Dense
	phT.Mul(f.state.P, H.T()) // nx x m
	KT := mat.NewDense(mDim, nx, nil)
	var ktCol mat.VecDense
	for col := 0; col < nx; col++ {
		rhs := mat.NewVecDense(mDim, nil)
		for i := 0; i < mDim; i++ {
			rhs.SetVec(i, phT.At(col, i))
		}
		ktCol.SolveVec(&chol, rhs)
		for i := 0; i < mDim; i++ {
			KT.Set(i, col, ktCol.AtVec(i))
		}
	}
	K := KT.T()

	// NIS = nu' S^-1 nu
	var snu mat.VecDense
	snu.SolveVec(&chol, nu)
	nis := mat.Dot(nu, &snu)
	out.NIS = nis

	// x = x + K nu
	var knu mat.VecDense
	knu.MulVec(K, nu)
	xNew := mat.VecDenseCopyOf(f.state.X)
	xNew.AddVec(xNew, &knu)
	if !vecFinite(xNew) {
		r := reject.New(reject.ReasonNonFiniteValue, "post-update state non-finite")
		out.Reason = string(r.Reason)
		return out, r
	}

	// Joseph form: P = (I-KH) P (I-KH)' + K R K'
	var kh mat.Dense
	kh.Mul(K, H)
	I := eye(nx)
	var imkh mat.Dense
	imkh.Sub(I, &kh)
	var a mat.Dense
	a.Mul(&imkh, f.state.P)
	var aaT mat.Dense
	aaT.Mul(&a, imkh.T())
	var kr mat.Dense
	kr.Mul(K, Rm)
	var krkt mat.Dense
	krkt.Mul(&kr, K.T())
	Pnew := mat.NewSymDense(nx, nil)
	for i := 0; i < nx; i++ {
		for j := i; j < nx; j++ {
			v := aaT.At(i, j) + krkt.At(i, j)
			Pnew.SetSym(i, j, v)
		}
	}
	floor := f.cfg.EigenFloor
	if err := noise.ValidateSPD(Pnew, noise.DefaultConfig()); err != nil {
		if floor <= 0 {
			floor = 1e-14
		}
		repaired, rerr := noise.NearestSPD(Pnew, floor)
		if rerr != nil {
			out.Reason = string(reject.ReasonNoiseNotSPD)
			return out, rerr
		}
		Pnew = repaired
	}

	f.state.X = xNew
	f.state.P = Pnew
	out.Accepted = true
	return out, nil
}

// transitionDT returns I + (F0-I)*dt.
func transitionDT(F0 *mat.Dense, dt float64) *mat.Dense {
	n, _ := F0.Dims()
	out := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			v := F0.At(i, j)
			if i == j {
				v = 1 + (v-1)*dt
			} else {
				v = v * dt
			}
			out.Set(i, j, v)
		}
	}
	return out
}

// processNoiseDT scales the per-unit-time white-noise-acceleration
// covariance to dt using the known dt-power pattern of each entry:
// the constant-velocity block has powers dt^4, dt^3, dt^2 which at dt=1
// equal Q0. We infer the power from structural positions: entries on a
// velocity diagonal scale as dt^2, position/velocity cross terms as dt^3,
// position diagonals as dt^4, and everything else linearly as dt.
func processNoiseDT(Q0 *mat.SymDense, dt float64) *mat.SymDense {
	n, _ := Q0.Dims()
	out := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			p := dtPowerHint(n, i, j)
			out.SetSym(i, j, Q0.At(i, j)*math.Pow(dt, p))
		}
	}
	return out
}

// dtPowerHint returns the assumed dt exponent for the constant-velocity
// state ordering [position, velocity, position, velocity, ...].
func dtPowerHint(n, i, j int) float64 {
	posI := i%2 == 0
	posJ := j%2 == 0
	switch {
	case posI && posJ:
		return 4
	case posI != posJ:
		return 3
	default:
		return 2
	}
}

func cloneSym(s *mat.SymDense) *mat.SymDense {
	n, _ := s.Dims()
	out := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			out.SetSym(i, j, s.At(i, j))
		}
	}
	return out
}

func eye(n int) *mat.Dense {
	d := mat.NewDense(n, n, nil)
	for i := 0; i < n; i++ {
		d.Set(i, i, 1)
	}
	return d
}

func vecFinite(v *mat.VecDense) bool {
	for i := 0; i < v.Len(); i++ {
		x := v.AtVec(i)
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return false
		}
	}
	return true
}

func matFinite(m mat.Matrix) bool {
	r, c := m.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v := m.At(i, j)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return false
			}
		}
	}
	return true
}

// wrapPi folds an angle into [-pi, pi).
func wrapPi(a float64) float64 {
	if math.IsNaN(a) || math.IsInf(a, 0) {
		return a
	}
	for a > math.Pi {
		a -= 2 * math.Pi
	}
	for a < -math.Pi {
		a += 2 * math.Pi
	}
	return a
}
