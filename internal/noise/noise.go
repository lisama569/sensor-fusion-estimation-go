// Package noise validates covariance matrices and provides numerical
// safeguards (symmetrization, symmetric positive-(semi)definite repair).
package noise

import (
	"math"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/reject"
	"gonum.org/v1/gonum/mat"
)

// Config controls validation tolerances.
type Config struct {
	// SPDTol is the relative negative-eigenvalue tolerance: a matrix is
	// accepted as positive semidefinite when its smallest eigenvalue is at
	// least -SPDTol*max(1, maxAbsEigenvalue). Matrices that are more
	// negative are rejected.
	SPDTol float64
	// SymTol is the relative asymmetry tolerance |A_ij-A_ji|.
	SymTol float64
	// EigenFloor is the absolute minimum eigenvalue applied when repairing.
	EigenFloor float64
}

// DefaultConfig returns sane default tolerances.
func DefaultConfig() Config {
	return Config{
		SPDTol:     1e-9,
		SymTol:     1e-10,
		EigenFloor: 1e-12,
	}
}

func finiteMatrix(S mat.Matrix) (int, bool) {
	r, c := S.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v := S.At(i, j)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return 0, false
			}
		}
	}
	return r, r == c
}

// ValidateFinite checks every entry of A is finite.
func ValidateFinite(A mat.Matrix) error {
	r, c := A.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v := A.At(i, j)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return reject.New(reject.ReasonNonFiniteValue,
					"matrix entry (%d,%d) is %v", i, j, v)
			}
		}
	}
	return nil
}

// ValidateSPD checks that S is square, finite, symmetric and symmetric
// positive semidefinite within tolerance. It returns a distinguishable
// rejection error (ReasonDimensionMismatch / ReasonNoiseNotSPD) otherwise.
//
// Semidefinite matrices (zero eigenvalues) are accepted: zero process or
// measurement noise on some directions is legal; a genuinely degenerate
// update (singular innovation covariance) is detected later at update time.
func ValidateSPD(S mat.Matrix, cfg Config) error {
	n, ok := finiteMatrix(S)
	if !ok {
		r, c := S.Dims()
		if r != c {
			return reject.New(reject.ReasonDimensionMismatch, "noise matrix must be square, got %dx%d", r, c)
		}
		return reject.New(reject.ReasonNoiseNotSPD, "noise matrix contains NaN/Inf")
	}

	// Symmetry check.
	var scale float64
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			a, b := S.At(i, j), S.At(j, i)
			if d := math.Abs(a - b); d > cfg.SymTol*math.Max(1, math.Max(math.Abs(a), math.Abs(b))) {
				return reject.New(reject.ReasonNoiseNotSPD,
					"matrix not symmetric at (%d,%d): %v vs %v", i, j, a, b)
			}
		}
		for j := 0; j < n; j++ {
			scale = math.Max(scale, math.Abs(S.At(i, j)))
		}
	}

	sym := asSymmetric(S, n)
	var es mat.EigenSym
	if ok := es.Factorize(sym, false); !ok {
		return reject.New(reject.ReasonNoiseNotSPD, "eigen decomposition failed")
	}
	vals := es.Values(nil)
	minEig, maxAbs := vals[0], math.Abs(vals[0])
	for _, v := range vals[1:] {
		minEig = math.Min(minEig, v)
		maxAbs = math.Max(maxAbs, math.Abs(v))
	}
	tol := cfg.SPDTol * math.Max(1, scale)
	if minEig < -tol {
		_ = maxAbs
		return reject.New(reject.ReasonNoiseNotSPD,
			"smallest eigenvalue %.3e below tolerance %.3e", minEig, -tol)
	}
	return nil
}

func asSymmetric(S mat.Matrix, n int) *mat.SymDense {
	if sy, ok := S.(*mat.SymDense); ok {
		return sy
	}
	sym := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			sym.SetSym(i, j, 0.5*(S.At(i, j)+S.At(j, i)))
		}
	}
	return sym
}

// Symmetrize returns 0.5*(A + A') as a SymDense.
func Symmetrize(A *mat.Dense) *mat.SymDense {
	r, c := A.Dims()
	if r != c {
		panic(reject.New(reject.ReasonDimensionMismatch, "Symmetrize requires square matrix, got %dx%d", r, c))
	}
	sym := mat.NewSymDense(r, nil)
	for i := 0; i < r; i++ {
		for j := i; j < r; j++ {
			sym.SetSym(i, j, 0.5*(A.At(i, j)+A.At(j, i)))
		}
	}
	return sym
}

// NearestSPD projects S onto the nearest symmetric positive semidefinite
// matrix (Higham 1988: symmetrize, clamp eigenvalues, symmetrize again)
// with eigenvalues floored at floor.
func NearestSPD(S *mat.SymDense, floor float64) (*mat.SymDense, error) {
	n, _ := S.Dims()
	var es mat.EigenSym
	if ok := es.Factorize(S, true); !ok {
		return nil, reject.New(reject.ReasonNoiseNotSPD, "eigen decomposition failed during repair")
	}
	vals := es.Values(nil)
	var V mat.Dense
	es.VectorsTo(&V)

	clamped := make([]float64, n)
	for i, v := range vals {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, reject.New(reject.ReasonNoiseNotSPD, "non-finite eigenvalue during repair")
		}
		clamped[i] = math.Max(v, floor)
	}
	dClamp := mat.NewDiagDense(n, clamped)
	// B = V * diag(clamped) * V'
	var vd mat.Dense
	vd.Mul(&V, dClamp)
	var b mat.Dense
	b.Mul(&vd, V.T())
	out := Symmetrize(&b)
	return out, nil
}
