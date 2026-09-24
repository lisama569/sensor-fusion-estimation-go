package mat

import "fmt"

// dimMismatch returns a dimension error.
func dimMismatch(op string, a, b *Matrix) error {
	return fmt.Errorf("mat: %s dimension mismatch %dx%d vs %dx%d",
		op, a.Rows, a.Cols, b.Rows, b.Cols)
}

// Add returns a+b.
func Add(a, b *Matrix) *Matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic(dimMismatch("add", a, b))
	}
	out := New(a.Rows, a.Cols)
	for i := range out.Data {
		out.Data[i] = a.Data[i] + b.Data[i]
	}
	return out
}

// Sub returns a-b.
func Sub(a, b *Matrix) *Matrix {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic(dimMismatch("sub", a, b))
	}
	out := New(a.Rows, a.Cols)
	for i := range out.Data {
		out.Data[i] = a.Data[i] - b.Data[i]
	}
	return out
}

// Scale returns a*s.
func Scale(a *Matrix, s float64) *Matrix {
	out := New(a.Rows, a.Cols)
	for i, v := range a.Data {
		out.Data[i] = v * s
	}
	return out
}

// Mul returns a*b.
func Mul(a, b *Matrix) *Matrix {
	if a.Cols != b.Rows {
		panic(dimMismatch("mul", a, b))
	}
	out := New(a.Rows, b.Cols)
	for i := 0; i < a.Rows; i++ {
		for k := 0; k < a.Cols; k++ {
			aik := a.Data[i*a.Cols+k]
			if aik == 0 {
				continue
			}
			rowOff := i * b.Cols
			bOff := k * b.Cols
			for j := 0; j < b.Cols; j++ {
				out.Data[rowOff+j] += aik * b.Data[bOff+j]
			}
		}
	}
	return out
}

// MulVec returns a*v where v is a column vector.
func MulVec(a *Matrix, v *Vector) *Vector {
	if a.Cols != v.Rows {
		panic(dimMismatch("mulvec", a, v))
	}
	out := New(a.Rows, 1)
	for i := 0; i < a.Rows; i++ {
		var s float64
		off := i * a.Cols
		for j := 0; j < a.Cols; j++ {
			s += a.Data[off+j] * v.Data[j]
		}
		out.Data[i] = s
	}
	return out
}

// T returns the transpose.
func T(a *Matrix) *Matrix {
	out := New(a.Cols, a.Rows)
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			out.Data[j*a.Rows+i] = a.Data[i*a.Cols+j]
		}
	}
	return out
}

// Transpose writes a^T into out. out must be a.Cols x a.Rows.
func Transpose(a, out *Matrix) {
	if out.Rows != a.Cols || out.Cols != a.Rows {
		panic("mat: Transpose target dimension mismatch")
	}
	for i := 0; i < a.Rows; i++ {
		for j := 0; j < a.Cols; j++ {
			out.Data[j*a.Rows+i] = a.Data[i*a.Cols+j]
		}
	}
}

// Equal reports whether two matrices match within tolerance.
func Equal(a, b *Matrix, tol float64) bool {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return false
	}
	for i := range a.Data {
		d := a.Data[i] - b.Data[i]
		if d < 0 {
			d = -d
		}
		if d > tol {
			return false
		}
	}
	return true
}
