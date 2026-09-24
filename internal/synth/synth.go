// Package synth generates reproducible local synthetic measurements for a
// 2D constant-velocity platform with state [x, vx, y, vy]:
//
//   - "gps":      linear position      z = [x, y] + v
//   - "velocity": linear velocity      z = [vx, vy] + v
//   - "polar":    nonlinear range/bearing from a fixed beacon
//     z = [hypot(x-bx, y-by), atan2(y-by, x-bx)] + v
//
// Every channel has its own sampling period and constant delay; the
// caller may shuffle and corrupt the produced sequence.
package synth

import (
	"math"
	"math/rand"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/models"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

// ChannelSpec describes one generated channel.
type ChannelSpec struct {
	ID      string
	Period  float64
	Delay   float64
	Noise2D [2][2]float64
	Kind    string // "gps" | "velocity" | "polar"
	Beacon  [2]float64
}

// Generator produces deterministic trajectories and measurements.
type Generator struct {
	rng    *rand.Rand
	period float64
	endT   float64
	x0     []float64
	chans  []ChannelSpec
	dists  map[string]*distmv.Normal
}

// NewGenerator constructs a generator with a fixed seed for reproducibility.
// period is the base trajectory tick; endT the horizon; x0 the initial
// [x, vx, y, vy].
func NewGenerator(seed int64, period, endT float64, x0 []float64, chans []ChannelSpec) *Generator {
	if len(x0) != 4 {
		panic("synth: x0 must be [x, vx, y, vy]")
	}
	g := &Generator{
		rng:    rand.New(rand.NewSource(seed)),
		period: period,
		endT:   endT,
		x0:     append([]float64(nil), x0...),
		chans:  append([]ChannelSpec(nil), chans...),
		dists:  make(map[string]*distmv.Normal),
	}
	for _, ch := range chans {
		cov := mat.NewSymDense(2, []float64{
			ch.Noise2D[0][0], ch.Noise2D[0][1],
			ch.Noise2D[1][0], ch.Noise2D[1][1],
		})
		// distmv.NewNormal requires SPD; fall back to a Cholesky from
		// repaired covariance so zero-variance specs still work.
		n, ok := distmv.NewNormal([]float64{0, 0}, cov, g.rng)
		if !ok {
			d := []float64{
				math.Max(cov.At(0, 0), 1e-12), 0,
				0, math.Max(cov.At(1, 1), 1e-12),
			}
			cov = mat.NewSymDense(2, d)
			n, _ = distmv.NewNormal([]float64{0, 0}, cov, g.rng)
		}
		g.dists[ch.ID] = n
	}
	return g
}

// GroundTruth returns the noiseless [x, vx, y, vy] at time t.
func (g *Generator) GroundTruth(t float64) []float64 {
	return []float64{
		g.x0[0] + g.x0[1]*t, g.x0[1],
		g.x0[2] + g.x0[3]*t, g.x0[3],
	}
}

// Sample generates the measurement list for every channel: each channel
// samples at its own period starting at t=Period. The timestamp is the
// true sampling time; the per-channel delay only affects arrival time
// (see Arrival), never the timestamp. Output is ordered by channel then
// time; use Shuffle to exercise out-of-order handling.
func (g *Generator) Sample() []models.Measurement {
	var out []models.Measurement
	for _, ch := range g.chans {
		for t := ch.Period; t <= g.endT+1e-9; t += ch.Period {
			gt := g.GroundTruth(t)
			noise := g.dists[ch.ID].Rand(nil)
			z := g.idealMeasurement(ch, gt, noise)
			out = append(out, models.Measurement{
				SensorID:  ch.ID,
				Timestamp: t,
				Z:         mat.NewVecDense(2, z),
			})
		}
	}
	return out
}

// Arrival returns the simulated arrival time of a measurement:
// sampling timestamp plus the channel's constant delay.
func (g *Generator) Arrival(mm models.Measurement) float64 {
	for _, ch := range g.chans {
		if ch.ID == mm.SensorID {
			return mm.Timestamp + ch.Delay
		}
	}
	return mm.Timestamp
}

func (g *Generator) idealMeasurement(ch ChannelSpec, gt, n []float64) []float64 {
	switch ch.Kind {
	case "gps":
		return []float64{gt[0] + n[0], gt[2] + n[1]}
	case "velocity":
		return []float64{gt[1] + n[0], gt[3] + n[1]}
	case "polar":
		dx := gt[0] - ch.Beacon[0]
		dy := gt[2] - ch.Beacon[1]
		return []float64{math.Hypot(dx, dy) + n[0], wrapPi(math.Atan2(dy, dx) + n[1])}
	default:
		panic("synth: unknown channel kind " + ch.Kind)
	}
}

// Corrupt replaces measurements from sensorID whose sampling time lies in
// [fromT, toT): when nonFinite is true they become NaN; otherwise a large
// diverging bias is added. Timestamps are left intact (so alignment rules
// still apply identically).
func (g *Generator) Corrupt(ms []models.Measurement, sensorID string, fromT, toT float64, nonFinite bool) []models.Measurement {
	for i := range ms {
		m := &ms[i]
		if m.SensorID != sensorID {
			continue
		}
		// Timestamps carry the delay; use the timestamp+delay window check
		// approximated directly against timestamp for deterministic tests.
		if m.Timestamp < fromT || m.Timestamp >= toT {
			continue
		}
		if nonFinite {
			m.Z = mat.NewVecDense(2, []float64{math.NaN(), math.NaN()})
		} else {
			m.Z = mat.NewVecDense(2, []float64{
				m.Z.AtVec(0) + 1e4,
				wrapPi(m.Z.AtVec(1) + math.Pi/2),
			})
		}
	}
	return ms
}

// Shuffle applies a deterministic Fisher-Yates permutation using the
// generator's RNG and returns the same slice for convenience.
func (g *Generator) Shuffle(ms []models.Measurement) []models.Measurement {
	g.rng.Shuffle(len(ms), func(i, j int) { ms[i], ms[j] = ms[j], ms[i] })
	return ms
}

// Models builds concrete measurement models for the given specs, given the
// 4D state. Polar models declare bearing row 1 as an angle.
func Models(ch ChannelSpec) (models.MeasureModel, error) {
	R := mat.NewSymDense(2, []float64{
		ch.Noise2D[0][0], ch.Noise2D[0][1],
		ch.Noise2D[1][0], ch.Noise2D[1][1],
	})
	switch ch.Kind {
	case "gps":
		H := mat.NewDense(2, 4, []float64{
			1, 0, 0, 0,
			0, 0, 1, 0,
		})
		return models.NewLinearModel(4, H, nil, R)
	case "velocity":
		H := mat.NewDense(2, 4, []float64{
			0, 1, 0, 0,
			0, 0, 0, 1,
		})
		return models.NewLinearModel(4, H, nil, R)
	case "polar":
		bx, by := ch.Beacon[0], ch.Beacon[1]
		eval := func(x *mat.VecDense) *mat.VecDense {
			px, py := x.AtVec(0), x.AtVec(2)
			dx, dy := px-bx, py-by
			return mat.NewVecDense(2, []float64{math.Hypot(dx, dy), math.Atan2(dy, dx)})
		}
		jac := func(x *mat.VecDense) *mat.Dense {
			px, py := x.AtVec(0), x.AtVec(2)
			dx, dy := px-bx, py-by
			r := math.Hypot(dx, dy)
			if r < 1e-9 {
				r = 1e-9
			}
			return mat.NewDense(2, 4, []float64{
				dx / r, 0, dy / r, 0,
				-dy / (r * r), 0, dx / (r * r), 0,
			})
		}
		return models.NewNonlinearModel(4, 2, eval, jac, nil, R, 1)
	default:
		panic("synth: unknown channel kind " + ch.Kind)
	}
}

func wrapPi(a float64) float64 {
	for a > math.Pi {
		a -= 2 * math.Pi
	}
	for a < -math.Pi {
		a += 2 * math.Pi
	}
	return a
}
