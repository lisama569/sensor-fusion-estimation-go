package sim

import (
	"testing"
)

func testCfg() Config {
	return Config{
		StartTime:       0,
		Duration:        4,
		Seed:            7,
		Shuffle:         true,
		InjectFaults:    true,
		FaultSource:     "gps-b",
		FaultStart:      1.0,
		FaultEnd:        2.0,
		EmitDuplicates:  true,
		DuplicateSource: "gps-a",
		Sensors: []SensorSpec{
			{Name: "gps-a", Kind: "position", RateHz: 5, Delay: 0.1, Jitter: 0.02, StdPos: 0.2},
			{Name: "gps-b", Kind: "position", RateHz: 4, Delay: 0.2, Jitter: 0.04, StdPos: 0.3},
			{Name: "radar", Kind: "rangebearing", RateHz: 2, Delay: 0.3,
				StdRange: 0.1, StdBearing: 0.01, Sx: -1, Sy: -1},
		},
	}
}

func TestGenerateDeterministic(t *testing.T) {
	a := Generate(testCfg())
	b := Generate(testCfg())
	if len(a.Samples) != len(b.Samples) || len(a.Truth) != len(b.Truth) {
		t.Fatal("dataset length differs")
	}
	for i := range a.Samples {
		x, y := a.Samples[i], b.Samples[i]
		if x.Source != y.Source || x.Time != y.Time ||
			len(x.Z) != len(y.Z) || x.NonFinite != y.NonFinite || x.Duplicate != y.Duplicate {
			t.Fatalf("sample %d differs between identical seeds", i)
		}
		for j := range x.Z {
			if x.Z[j] != y.Z[j] {
				t.Fatalf("sample %d z differs", i)
			}
		}
	}
}

func TestGenerateContainsFaultsAndDuplicates(t *testing.T) {
	ds := Generate(testCfg())
	nf, dup := 0, 0
	for _, s := range ds.Samples {
		if s.NonFinite {
			nf++
		}
		if s.Duplicate {
			dup++
		}
	}
	if nf == 0 {
		t.Fatal("expected injected non-finite samples")
	}
	if dup == 0 {
		t.Fatal("expected an emitted duplicate")
	}
}

func TestDifferentSeedDifferentShuffle(t *testing.T) {
	c1, c2 := testCfg(), testCfg()
	c2.Seed = 8
	a := Generate(c1)
	b := Generate(c2)
	diff := false
	for i := range a.Samples {
		if a.Samples[i].Source != b.Samples[i].Source || a.Samples[i].Time != b.Samples[i].Time {
			diff = true
			break
		}
	}
	if !diff {
		t.Fatal("different seeds unexpectedly produced identical emission order")
	}
}
