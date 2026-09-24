package noise

import (
	"math"
	"testing"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/reject"
	"gonum.org/v1/gonum/mat"
)

func TestValidateSPD(t *testing.T) {
	cfg := DefaultConfig()

	good := mat.NewSymDense(2, []float64{2, 0.5, 0, 1})
	if err := ValidateSPD(good, cfg); err != nil {
		t.Fatalf("SPD matrix rejected: %v", err)
	}
	t.Log("decision: [[2,0.5],[0.5,1]] accepted as SPD")

	// Semidefinite is legal (zero noise direction).
	semi := mat.NewSymDense(2, []float64{1, 0, 0, 0})
	if err := ValidateSPD(semi, cfg); err != nil {
		t.Fatalf("PSD semidefinite matrix rejected: %v", err)
	}
	t.Log("decision: diag(1,0) accepted as PSD (zero noise direction legal)")

	bad := mat.NewSymDense(2, []float64{1, 0, 0, -1})
	err := ValidateSPD(bad, cfg)
	if err == nil || reject.Of(err) != reject.ReasonNoiseNotSPD {
		t.Fatalf("negative eigenvalue must give %s, got %v", reject.ReasonNoiseNotSPD, err)
	}
	t.Logf("decision: diag(1,-1) rejected: %v", err)

	// Asymmetry beyond tolerance.
	asym := mat.NewDense(2, 2, []float64{1, 1, 0, 1})
	err = ValidateSPD(asym, cfg)
	if err == nil || reject.Of(err) != reject.ReasonNoiseNotSPD {
		t.Fatalf("asymmetric matrix must give %s, got %v", reject.ReasonNoiseNotSPD, err)
	}
	t.Logf("decision: [[1,1],[0,1]] rejected: %v", err)

	// Non-finite.
	nf := mat.NewSymDense(2, []float64{1, 0, 0, math.NaN()})
	err = ValidateSPD(nf, cfg)
	if err == nil || reject.Of(err) != reject.ReasonNoiseNotSPD {
		t.Fatalf("NaN matrix must give %s, got %v", reject.ReasonNoiseNotSPD, err)
	}
	t.Logf("decision: NaN covariance rejected: %v", err)

	// Non-square -> dimension mismatch.
	rect := mat.NewDense(2, 3, nil)
	err = ValidateSPD(rect, cfg)
	if err == nil || reject.Of(err) != reject.ReasonDimensionMismatch {
		t.Fatalf("rectangular matrix must give %s, got %v", reject.ReasonDimensionMismatch, err)
	}
	t.Logf("decision: 2x3 matrix rejected: %v", err)
}

func TestNearestSPD(t *testing.T) {
	// Indefinite matrix gets its negative eigenvalue floored.
	S := mat.NewSymDense(2, []float64{1, 0, 0, -2})
	out, err := NearestSPD(S, 1e-10)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateSPD(out, DefaultConfig()); err != nil {
		t.Fatalf("repaired matrix not PSD: %v", err)
	}
	// Eigenvalues are now exactly {1, 1e-10}.
	var es mat.EigenSym
	es.Factorize(out, false)
	vals := es.Values(nil)
	for _, v := range vals {
		if v < 0 {
			t.Fatalf("negative eigenvalue after repair: %v", vals)
		}
	}
	t.Logf("decision: eig(diag(1,-2)) repaired to %v (all >= 0)", vals)

	// Tiny negative eigenvalue from roundoff is repairable too.
	rnd := mat.NewSymDense(2, []float64{1, 0, 0, -1e-14})
	out2, err := NearestSPD(rnd, 1e-12)
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateSPD(out2, DefaultConfig()); err != nil {
		t.Fatalf("roundoff repair failed: %v", err)
	}
	t.Log("decision: -1e-14 eigenvalue repaired within tolerance")
}
