package mat

import (
	"errors"
	"math"
	"sort"
)

// Finite reports that every element is a finite real number (not NaN/Inf).
func Finite(m *Matrix) bool {
	for _, v := range m.Data {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return false
		}
	}
	return true
}

// EigenSymmetric returns the eigenvalues of a symmetric matrix in ascending
// order using the cyclic Jacobi eigenvalue method.
func EigenSymmetric(m *Matrix) ([]float64, error) {
	if m.Rows != m.Cols {
		return nil, errors.New("mat.EigenSymmetric: non-square matrix")
	}
	n := m.Rows
	a := Symmetrize(m)
	const maxSweeps = 100
	const tol = 1e-14
	for sweep := 0; sweep < maxSweeps; sweep++ {
		off := 0.0
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				off += a.At(p, q) * a.At(p, q)
			}
		}
		if math.Sqrt(off) <= tol {
			break
		}
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				apq := a.At(p, q)
				if math.Abs(apq) < 1e-300 {
					continue
				}
				app := a.At(p, p)
				aqq := a.At(q, q)
				var tau, t, c, s float64
				if math.Abs(app-aqq) < 1e-300 {
					t = 1
				} else {
					tau = (aqq - app) / (2 * apq)
					t = sign(1, tau) / (math.Abs(tau) + math.Sqrt(1+tau*tau))
				}
				c = 1 / math.Sqrt(1+t*t)
				s = t * c
				a.Set(p, p, app-t*apq)
				a.Set(q, q, aqq+t*apq)
				a.Set(p, q, 0)
				a.Set(q, p, 0)
				for r := 0; r < n; r++ {
					if r == p || r == q {
						continue
					}
					arq := a.At(r, q)
					arp := a.At(r, p)
					a.Set(r, p, c*arp-s*arq)
					a.Set(p, r, a.At(r, p))
					a.Set(r, q, s*arp+c*arq)
					a.Set(q, r, a.At(r, q))
				}
			}
		}
	}
	eig := make([]float64, n)
	for i := 0; i < n; i++ {
		eig[i] = a.At(i, i)
	}
	sort.Float64s(eig)
	return eig, nil
}

func sign(a, b float64) float64 {
	if b < 0 {
		return -a
	}
	return a
}

// IsPSD reports whether a symmetric matrix is positive semidefinite:
// every eigenvalue >= -tol*max(1,|lambda_max|).
func IsPSD(m *Matrix, tol float64) (bool, error) {
	eig, err := EigenSymmetric(m)
	if err != nil {
		return false, err
	}
	var scale float64 = 1
	for _, l := range eig {
		if a := math.Abs(l); a > scale {
			scale = a
		}
	}
	limit := -tol * scale
	for _, l := range eig {
		if l < limit {
			return false, nil
		}
	}
	return true, nil
}

// MinEig returns the smallest eigenvalue of a symmetric matrix.
func MinEig(m *Matrix) (float64, error) {
	eig, err := EigenSymmetric(m)
	if err != nil {
		return 0, err
	}
	return eig[0], nil
}

// NormFrob returns the Frobenius norm.
func NormFrob(m *Matrix) float64 {
	var s float64
	for _, v := range m.Data {
		s += v * v
	}
	return math.Sqrt(s)
}
