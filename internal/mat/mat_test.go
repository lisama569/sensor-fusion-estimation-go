package mat

import (
	"math"
	"testing"
)

func TestMulAndTranspose(t *testing.T) {
	a := From([][]float64{{1, 2}, {3, 4}, {5, 6}})
	b := From([][]float64{{7, 8, 9}, {10, 11, 12}})
	c := Mul(a, b)
	want := [][]float64{{27, 30, 33}, {61, 68, 75}, {95, 106, 117}}
	if !Equal(c, From(want), 1e-12) {
		t.Fatalf("Mul wrong: %v", c.Data)
	}
	if !Equal(T(b), From([][]float64{{7, 10}, {8, 11}, {9, 12}}), 1e-12) {
		t.Fatalf("T wrong")
	}
}

func TestLDRoundTrip(t *testing.T) {
	a := From([][]float64{{4, 1}, {1, 3}})
	c, err := LDL(a)
	if err != nil {
		t.Fatal(err)
	}
	LDLt := Mul(Mul(c.L, diagMat(c.D)), T(c.L))
	if !Equal(LDLt, a, 1e-12) {
		t.Fatalf("LDL^T roundtrip wrong: %v", LDLt.Data)
	}
	x := c.Solve(Vec(1, 2))
	got := MulVec(a, x)
	if !Equal(got, Vec(1, 2), 1e-12) {
		t.Fatalf("Solve wrong: %v", got.Data)
	}
	inv, err := InvertSPD(a)
	if err != nil {
		t.Fatal(err)
	}
	prod := Mul(a, inv)
	if !Equal(prod, Identity(2), 1e-10) {
		t.Fatalf("inverse wrong: %v", prod.Data)
	}
}

func TestEigenAndPSD(t *testing.T) {
	// eigenvalues of [[2,1],[1,2]] are 1 and 3
	a := From([][]float64{{2, 1}, {1, 2}})
	eig, err := EigenSymmetric(a)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(eig[0]-1) > 1e-10 || math.Abs(eig[1]-3) > 1e-10 {
		t.Fatalf("eig wrong: %v", eig)
	}
	ok, _ := IsPSD(a, 1e-10)
	if !ok {
		t.Fatal("should be PSD")
	}
	bad := From([][]float64{{1, 0}, {0, -0.1}})
	ok, _ = IsPSD(bad, 1e-10)
	if ok {
		t.Fatal("negative eigenvalue should not be PSD")
	}
	if _, err := LDL(bad); err == nil {
		t.Fatal("LDL must reject indefinite matrix")
	}
}

func diagMat(d []float64) *Matrix {
	m := New(len(d), len(d))
	for i, v := range d {
		m.Set(i, i, v)
	}
	return m
}
