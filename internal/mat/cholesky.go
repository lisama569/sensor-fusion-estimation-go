package mat

import (
	"errors"
	"math"
)

// Cholesky holds an LDL^T decomposition: A = L D L^T with L unit lower
// triangular and D diagonal.
type Cholesky struct {
	N int
	L *Matrix
	D []float64
}

// ErrNotSPD indicates the matrix is not symmetric positive definite within
// the working tolerance.
var ErrNotSPD = errors.New("mat: matrix is not symmetric positive definite")

// LDL returns the LDL^T decomposition of a symmetric matrix. If the matrix
// has a non-positive pivot (scaled for magnitude) ErrNotSPD is returned.
func LDL(a *Matrix) (*Cholesky, error) {
	if a.Rows != a.Cols {
		return nil, errors.New("mat.LDL: non-square matrix")
	}
	n := a.Rows
	// Symmetrize defensively against tiny asymmetric round-off.
	s := Symmetrize(a)
	L := Identity(n)
	d := make([]float64, n)

	var scale float64 = 1
	for _, v := range s.Data {
		if x := absv(v); x > scale {
			scale = x
		}
	}
	// Pivots smaller than this are treated as zero/negative for stability.
	pivotFloor := math.Max(1e-14*scale, 1e-300)

	for j := 0; j < n; j++ {
		var diag float64 = s.At(j, j)
		for k := 0; k < j; k++ {
			lik := L.At(j, k)
			diag -= lik * lik * d[k]
		}
		if math.IsNaN(diag) || math.IsInf(diag, 0) || diag <= pivotFloor {
			return nil, ErrNotSPD
		}
		d[j] = diag
		for i := j + 1; i < n; i++ {
			var v float64 = s.At(i, j)
			for k := 0; k < j; k++ {
				v -= L.At(i, k) * L.At(j, k) * d[k]
			}
			L.Set(i, j, v/diag)
		}
	}
	return &Cholesky{N: n, L: L, D: d}, nil
}

// Solve returns x such that A x = b, using A = L D L^T.
func (c *Cholesky) Solve(b *Vector) *Vector {
	if b.Rows != c.N {
		panic("mat: Cholesky.Solve dimension mismatch")
	}
	n := c.N
	// Forward: L y = b
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		v := b.Data[i]
		for k := 0; k < i; k++ {
			v -= c.L.At(i, k) * y[k]
		}
		y[i] = v
	}
	// Diagonal: D z = y
	z := make([]float64, n)
	for i := 0; i < n; i++ {
		z[i] = y[i] / c.D[i]
	}
	// Back: L^T x = z
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		v := z[i]
		for k := i + 1; k < n; k++ {
			v -= c.L.At(k, i) * x[k]
		}
		x[i] = v
	}
	return AsVec(x)
}

// SolveSPD returns A^{-1} b for symmetric positive definite A.
func SolveSPD(a *Matrix, b *Vector) (*Vector, error) {
	c, err := LDL(a)
	if err != nil {
		return nil, err
	}
	return c.Solve(b), nil
}

// InvertSPD returns the inverse of a symmetric positive definite matrix.
func InvertSPD(a *Matrix) (*Matrix, error) {
	c, err := LDL(a)
	if err != nil {
		return nil, err
	}
	n := a.Rows
	inv := New(n, n)
	e := make([]float64, n)
	for j := 0; j < n; j++ {
		for i := range e {
			e[i] = 0
		}
		e[j] = 1
		col := c.Solve(AsVec(e))
		for i := 0; i < n; i++ {
			inv.Set(i, j, col.Data[i])
		}
	}
	// Force exact symmetry of the result.
	return Symmetrize(inv), nil
}
