// Package sim generates deterministic local measurement datasets from a
// scripted ground-truth trajectory, simulating multiple sensors with
// different rates, delay, jitter, duplicates and injected faults.
package sim

import (
	"math"
	"math/rand"
	"sort"
)

// TruthPoint is one ground-truth sample: time and [px, py, vx, vy].
type TruthPoint struct {
	Time           float64
	Px, Py, Vx, Vy float64
}

// Sample is one generated raw measurement: time and [z0, z1].
type Sample struct {
	Time   float64
	Source string
	Z      []float64
	// NonFinite marks a sample whose value should be corrupted to NaN/Inf to
	// simulate a hardware fault.
	NonFinite bool
	// Duplicate marks a repeated emission of the same (source, time).
	Duplicate bool
}

// SensorSpec describes one simulated sensor stream.
type SensorSpec struct {
	Name       string
	Kind       string // "position" (linear) or "rangebearing" (nonlinear)
	RateHz     float64
	Delay      float64
	Jitter     float64
	StdPos     float64 // position measurement noise std (both components)
	StdRange   float64
	StdBearing float64
	Sx, Sy     float64 // sensor origin for rangebearing
}

// Dataset is a generated scenario.
type Dataset struct {
	Truth   []TruthPoint
	Samples []Sample
}

// Config controls deterministic generation.
type Config struct {
	StartTime float64
	Duration  float64
	Sensors   []SensorSpec
	Seed      int64
	// InjectFaults emits non-finite/diverging samples for a time window on
	// FaultSource, then emits healthy samples again to exercise recovery.
	InjectFaults bool
	FaultSource  string
	FaultStart   float64
	FaultEnd     float64
	// EmitDuplicates repeats one sample on one source.
	EmitDuplicates  bool
	DuplicateSource string
	// Shuffle permutes emission order (timestamps stay unchanged) to
	// demonstrate out-of-order handling.
	Shuffle bool
}

// truthAt evaluates the scripted trajectory (constant heading speed with a
// gentle turn) at time t.
func truthAt(t float64) TruthPoint {
	// Trajectory: start at origin moving +x at 1 m/s, with a slowly turning
	// heading omega = 0.15 rad/s, turn radius style motion.
	const speed = 1.0
	const omega = 0.15
	theta := omega * t
	// Integrate velocity (cos,sin)*speed:
	// x = speed/omega sin(theta), y = speed/omega (1-cos(theta)).
	px := speed / omega * math.Sin(theta)
	py := speed / omega * (1 - math.Cos(theta))
	vx := speed * math.Cos(theta)
	vy := speed * math.Sin(theta)
	return TruthPoint{Time: t, Px: px, Py: py, Vx: vx, Vy: vy}
}

// Generate builds the deterministic dataset. A dedicated rand source seeded
// from Config.Seed makes every run reproduce the exact same sequence.
func Generate(cfg Config) *Dataset {
	rng := rand.New(rand.NewSource(cfg.Seed))
	ds := &Dataset{}

	// Ground truth at a fine 20 Hz grid.
	const truthHz = 20
	end := cfg.StartTime + cfg.Duration
	for t := cfg.StartTime; t <= end+1e-9; t += 1.0 / truthHz {
		ds.Truth = append(ds.Truth, truthAt(t))
	}

	for _, sen := range cfg.Sensors {
		if sen.RateHz <= 0 {
			continue
		}
		period := 1.0 / sen.RateHz
		for k := 0; ; k++ {
			raw := cfg.StartTime + float64(k)*period
			if raw > end+1e-9 {
				break
			}
			tr := truthAt(raw)
			jit := 0.0
			if sen.Jitter > 0 {
				jit = (rng.Float64()*2 - 1) * sen.Jitter
			}
			// Emission time = sample time + delay + jitter; the carried
			// timestamp stays the true measurement time, so delayed data
			// arrives "late" relative to wall clock.
			emit := raw + sen.Delay + jit
			if emit > end+sen.Delay+1e-9 {
				// keep anyway to make delays visible within scenario
			}

			z := measure(sen, tr, rng)
			inFaultWindow := cfg.InjectFaults && sen.Name == cfg.FaultSource &&
				raw >= cfg.FaultStart && raw < cfg.FaultEnd
			s := Sample{Time: raw, Source: sen.Name, Z: z, NonFinite: inFaultWindow}
			ds.Samples = append(ds.Samples, s)

			if cfg.EmitDuplicates && sen.Name == cfg.DuplicateSource && k == 2 {
				dup := Sample{Time: raw, Source: sen.Name, Z: append([]float64(nil), z...), Duplicate: true}
				ds.Samples = append(ds.Samples, dup)
			}
		}
	}

	// Default order sorts by emission time so delayed sensors come later;
	// Shuffle permutes to simulate concurrent unsynchronized arrival.
	if cfg.Shuffle {
		rng.Shuffle(len(ds.Samples), func(i, j int) {
			ds.Samples[i], ds.Samples[j] = ds.Samples[j], ds.Samples[i]
		})
	} else {
		sort.SliceStable(ds.Samples, func(i, j int) bool {
			return ds.Samples[i].Time < ds.Samples[j].Time
		})
	}
	return ds
}

func measure(sen SensorSpec, tr TruthPoint, rng *rand.Rand) []float64 {
	switch sen.Kind {
	case "rangebearing":
		dx := tr.Px - sen.Sx
		dy := tr.Py - sen.Sy
		r := math.Hypot(dx, dy) + rng.NormFloat64()*sen.StdRange
		b := math.Atan2(dy, dx) + rng.NormFloat64()*sen.StdBearing
		return []float64{r, b}
	default: // position
		x := tr.Px + rng.NormFloat64()*sen.StdPos
		y := tr.Py + rng.NormFloat64()*sen.StdPos
		return []float64{x, y}
	}
}

// CorruptForFault replaces a sample's values with non-finite markers to
// emulate a hardware failure. The first faulty sample uses NaN; later ones
// use +Inf so detection paths for both are exercised.
func CorruptForFault(s *Sample, index int) {
	if index%2 == 0 {
		s.Z = []float64{math.NaN(), 0}
	} else {
		s.Z = []float64{math.Inf(1), math.Inf(-1)}
	}
}
