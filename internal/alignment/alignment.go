// Package alignment time-aligns multi-rate, delayed, out-of-order and
// duplicated measurements using a monotone watermark.
//
// Rules (see docs/ARCHITECTURE.md):
//
//   - Every measurement belongs to one sensor channel and carries an
//     absolute timestamp.
//   - Accept(t) rejects: non-finite timestamps; duplicates, when the exact
//     (sensor, timestamp) pair was already accepted; and stale inputs,
//     when timestamp < watermark - MaxDelay. Rejections are explicit.
//   - AdvanceTo(t) releases all queued measurements with timestamp <= t
//     in the deterministic order (timestamp, sensorID, arrivalSeq); the
//     watermark then becomes t and never moves backwards.
//   - Flush releases everything still queued.
package alignment

import (
	"math"
	"sort"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/models"
	"github.com/lisama569/sensor-fusion-estimation-go/internal/reject"
)

// Config configures the alignment buffer.
type Config struct {
	// MaxDelay is the maximum tolerated inter-arrival delay. Measurements
	// older than watermark-MaxDelay are stale and rejected.
	MaxDelay float64
}

type entry struct {
	mm  models.Measurement
	seq int64
}

// Buffer holds accepted-but-not-yet-released measurements and tracks, for
// every sensor channel, the set of timestamps ever accepted (duplicate
// detection) and the global monotone watermark.
type Buffer struct {
	cfg       Config
	queued    []entry
	seen      map[string]map[float64]struct{}
	watermark float64
	seqGen    int64
}

// NewBuffer constructs an empty buffer. Non-positive MaxDelay is rejected
// at the fuser layer; here a zero value means "no delay tolerated".
func NewBuffer(cfg Config) *Buffer {
	return &Buffer{
		cfg:       cfg,
		seen:      make(map[string]map[float64]struct{}),
		watermark: math.Inf(-1),
	}
}

// Accept validates and enqueues one measurement. It explicitly rejects bad
// timestamps (ReasonBadTimestamp), duplicates (ReasonDuplicateTimestamp)
// and stale inputs (ReasonStaleTimestamp) instead of dropping them.
// Rejected inputs are never stored.
func (b *Buffer) Accept(mm models.Measurement) error {
	if math.IsNaN(mm.Timestamp) || math.IsInf(mm.Timestamp, 0) {
		return reject.New(reject.ReasonBadTimestamp,
			"sensor %q submitted non-finite timestamp %v", mm.SensorID, mm.Timestamp)
	}
	if mm.Z == nil {
		return reject.New(reject.ReasonNonFiniteValue,
			"sensor %q at t=%v submitted nil measurement vector", mm.SensorID, mm.Timestamp)
	}
	for i := 0; i < mm.Z.Len(); i++ {
		v := mm.Z.AtVec(i)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return reject.New(reject.ReasonNonFiniteValue,
				"sensor %q at t=%v has non-finite component z[%d]=%v",
				mm.SensorID, mm.Timestamp, i, v)
		}
	}

	set := b.seen[mm.SensorID]
	if set == nil {
		set = make(map[float64]struct{})
		b.seen[mm.SensorID] = set
	}
	if _, dup := set[mm.Timestamp]; dup {
		return reject.New(reject.ReasonDuplicateTimestamp,
			"sensor %q already accepted timestamp %v", mm.SensorID, mm.Timestamp)
	}

	if !math.IsInf(b.watermark, -1) && mm.Timestamp < b.watermark-b.cfg.MaxDelay {
		return reject.New(reject.ReasonStaleTimestamp,
			"sensor %q timestamp %.6f is older than watermark %.6f minus max delay %.6f",
			mm.SensorID, mm.Timestamp, b.watermark, b.cfg.MaxDelay)
	}

	set[mm.Timestamp] = struct{}{}
	b.seqGen++
	b.queued = append(b.queued, entry{mm: mm, seq: b.seqGen})
	return nil
}

// AdvanceTo moves the watermark to t (it must not move backwards) and
// releases every queued measurement with timestamp <= t, ordered
// deterministically by (timestamp, sensorID, arrivalSeq).
func (b *Buffer) AdvanceTo(t float64) []models.Measurement {
	if math.IsNaN(t) {
		return nil
	}
	if !math.IsInf(b.watermark, -1) && t < b.watermark {
		t = b.watermark
	}

	var ready, keep []entry
	for _, e := range b.queued {
		if e.mm.Timestamp <= t {
			ready = append(ready, e)
		} else {
			keep = append(keep, e)
		}
	}
	b.queued = keep

	sort.SliceStable(ready, func(i, j int) bool {
		if ready[i].mm.Timestamp != ready[j].mm.Timestamp {
			return ready[i].mm.Timestamp < ready[j].mm.Timestamp
		}
		if ready[i].mm.SensorID != ready[j].mm.SensorID {
			return ready[i].mm.SensorID < ready[j].mm.SensorID
		}
		return ready[i].seq < ready[j].seq
	})

	out := make([]models.Measurement, len(ready))
	for i, e := range ready {
		out[i] = e.mm
	}
	b.watermark = t
	return out
}

// Flush releases everything still queued, advancing the watermark past all
// buffered timestamps. It returns ReasonNoData if nothing is queued.
func (b *Buffer) Flush() ([]models.Measurement, error) {
	if len(b.queued) == 0 {
		return nil, reject.New(reject.ReasonNoData, "flush called with empty buffer")
	}
	latest := math.Inf(-1)
	for _, e := range b.queued {
		if e.mm.Timestamp > latest {
			latest = e.mm.Timestamp
		}
	}
	out := b.AdvanceTo(latest)
	return out, nil
}

// Watermark returns the current global watermark (-Inf before any advance).
func (b *Buffer) Watermark() float64 { return b.watermark }

// Pending returns the number of queued measurements.
func (b *Buffer) Pending() int { return len(b.queued) }

// SeenCount reports how many timestamps have ever been accepted on a
// channel (diagnostics/testing).
func (b *Buffer) SeenCount(sensorID string) int { return len(b.seen[sensorID]) }
