package mat

import "fmt"

// Symmetrize returns 0.5*(m+m^T). The input must be square.
func Symmetrize(m *Matrix) *Matrix {
	if m.Rows != m.Cols {
		panic(fmt.Sprintf("mat.Symmetrize: non-square %dx%d", m.Rows, m.Cols))
	}
	n := m.Rows
	out := New(n, n)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			v := 0.5 * (m.At(i, j) + m.At(j, i))
			out.Set(i, j, v)
			out.Set(j, i, v)
		}
	}
	return out
}

// IsSymmetric reports whether m is square and |m_ij-m_ji| <= tol
// (relative to the matrix scale; tol is absolute at scale 1).
func IsSymmetric(m *Matrix, tol float64) bool {
	if m.Rows != m.Cols {
		return false
	}
	n := m.Rows
	var scale float64 = 1
	for _, v := range m.Data {
		if a := absv(v); a > scale {
			scale = a
		}
	}
	limit := tol * scale
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			d := m.At(i, j) - m.At(j, i)
			if absv(d) > limit {
				return false
			}
		}
	}
	return true
}

func absv(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
