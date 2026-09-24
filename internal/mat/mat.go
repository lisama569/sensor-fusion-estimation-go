// Package mat provides a small dependency-free dense matrix toolkit used by the
// state-estimation engine. All functions operate on float64 data stored in
// row-major order.
package mat

import "fmt"

// Matrix is a dense row-major matrix.
type Matrix struct {
	Rows int
	Cols int
	Data []float64
}

// New returns a zero matrix with the given dimensions. Negative or zero
// dimensions are rejected.
func New(rows, cols int) *Matrix {
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("mat.New: invalid dimensions %dx%d", rows, cols))
	}
	return &Matrix{Rows: rows, Cols: cols, Data: make([]float64, rows*cols)}
}

// Identity returns an n-by-n identity matrix.
func Identity(n int) *Matrix {
	m := New(n, n)
	for i := 0; i < n; i++ {
		m.Data[i*n+i] = 1
	}
	return m
}

// From builds a matrix from nested row slices. All rows must share length.
func From(rows [][]float64) *Matrix {
	if len(rows) == 0 {
		panic("mat.From: no rows")
	}
	cols := len(rows[0])
	if cols == 0 {
		panic("mat.From: zero columns")
	}
	m := New(len(rows), cols)
	for r, row := range rows {
		if len(row) != cols {
			panic("mat.From: ragged rows")
		}
		copy(m.Data[r*cols:], row)
	}
	return m
}

// At returns the element at (r, c).
func (m *Matrix) At(r, c int) float64 {
	return m.Data[r*m.Cols+c]
}

// Set writes the element at (r, c).
func (m *Matrix) Set(r, c int, v float64) {
	m.Data[r*m.Cols+c] = v
}

// Clone returns an independent copy.
func (m *Matrix) Clone() *Matrix {
	d := make([]float64, len(m.Data))
	copy(d, m.Data)
	return &Matrix{Rows: m.Rows, Cols: m.Cols, Data: d}
}

// Slice returns row r as a copied slice.
func (m *Matrix) Slice(r int) []float64 {
	out := make([]float64, m.Cols)
	copy(out, m.Data[r*m.Cols:(r+1)*m.Cols])
	return out
}

// Vector is a column vector stored in a single-column matrix.
type Vector = Matrix

// Vec builds a column vector.
func Vec(values ...float64) *Vector {
	if len(values) == 0 {
		panic("mat.Vec: empty")
	}
	v := New(len(values), 1)
	copy(v.Data, values)
	return v
}

// AsVec wraps a slice as a length-n column vector without copying.
func AsVec(values []float64) *Vector {
	if len(values) == 0 {
		panic("mat.AsVec: empty")
	}
	return &Matrix{Rows: len(values), Cols: 1, Data: values}
}

// VecAt returns the i-th element of a column vector.
func VecAt(v *Vector, i int) float64 { return v.Data[i] }
