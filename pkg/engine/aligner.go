package engine

import (
	"math"
	"sort"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/model"
)

// Measurement is one timestamped sample offered to the engine.
type Measurement struct {
	Time   float64
	Source string
	Z      *mat.Vector
	Model  model.MeasurementModel
}

// aligner buffers submitted measurements and, when time advances, partitions
// them deterministically into DUE (fusable) and REJECTED (expired/duplicate).
// Every accept/reject decision is made inside advance() in (time, source)
// order, so concurrent submissions of the identical multiset always converge
// to the exact same result.
type aligner struct {
	now     float64
	horizon float64
	pending []*Measurement
	// committed is the highest timestamp already released per source.
	committed map[string]float64
}

// newAligner creates an aligner starting at time start with a delay horizon.
func newAligner(start, horizon float64) *aligner {
	return &aligner{
		now:       start,
		horizon:   horizon,
		pending:   make([]*Measurement, 0, 64),
		committed: make(map[string]float64),
	}
}

// submit only buffers the measurement. Order-dependent decisions (expired /
// duplicate) are deferred to advance so they never depend on arrival order.
func (a *aligner) submit(m *Measurement) error {
	if m == nil {
		return reject("", 0, ReasonDimensionMismatch, "nil measurement")
	}
	if math.IsNaN(m.Time) || math.IsInf(m.Time, 0) {
		// A NaN/Inf timestamp cannot be ordered; reject immediately, which is
		// independent of arrival order.
		return reject(m.Source, m.Time, ReasonBadTimestamp, "timestamp is NaN/Inf")
	}
	a.pending = append(a.pending, m)
	return nil
}

// advance moves the committed time to t (never backwards) and partitions the
// buffered measurements with time <= t into due (fusable) and rejected. It
// processes candidates in sorted (time, source) order and applies, in
// priority:
//  1. expired: time < lastCommitted(source) - horizon (time no longer usable);
//  2. duplicate: exact (source, time) already committed, or repeated within
//     this same batch;
//  3. otherwise the sample is due, including a legal earlier-but-in-horizon
//     late arrival.
func (a *aligner) advance(t float64) (due []*Measurement, rejected []*RejectError) {
	if t < a.now {
		t = a.now
	}
	cand := make([]*Measurement, 0, len(a.pending))
	future := make([]*Measurement, 0, len(a.pending))
	for _, m := range a.pending {
		if m.Time <= t {
			cand = append(cand, m)
		} else {
			future = append(future, m)
		}
	}
	a.pending = future

	sort.SliceStable(cand, func(i, j int) bool {
		if cand[i].Time != cand[j].Time {
			return cand[i].Time < cand[j].Time
		}
		return cand[i].Source < cand[j].Source
	})

	// Count occurrences per (source, timestamp) within this batch. A key with
	// more than one candidate is a duplicate group: ALL of its members are
	// rejected, because the pipeline cannot tell which copy is authoritative.
	// This rule is independent of the order in which goroutines delivered the
	// samples, which is essential for concurrency reproducibility.
	type dupKey struct {
		source string
		time   float64
	}
	count := make(map[dupKey]int, len(cand))
	for _, m := range cand {
		count[dupKey{m.Source, m.Time}]++
	}

	batchSeen := make(map[string]map[float64]struct{})
	for _, m := range cand {
		// Rule 1: expired takes precedence (the timestamp itself is stale).
		if m.Time < t-a.horizon {
			rejected = append(rejected, reject(m.Source, m.Time, ReasonExpired,
				"timestamp older than committed time minus delay horizon"))
			continue
		}
		// Rule 2: any member of a duplicate group is refused.
		if count[dupKey{m.Source, m.Time}] > 1 {
			rejected = append(rejected, reject(m.Source, m.Time, ReasonDuplicate,
				"same source submitted more than one measurement at this exact timestamp; all copies refused"))
			continue
		}
		// Defensive duplicate guard against an identical key already seen.
		seen := batchSeen[m.Source]
		if seen == nil {
			seen = make(map[float64]struct{})
			batchSeen[m.Source] = seen
		}
		if _, dup := seen[m.Time]; dup {
			rejected = append(rejected, reject(m.Source, m.Time, ReasonDuplicate,
				"same source submitted a measurement at this exact timestamp more than once"))
			continue
		}
		seen[m.Time] = struct{}{}

		// Rule 3: due (including in-horizon late arrivals).
		due = append(due, m)
		if last, ok := a.committed[m.Source]; !ok || m.Time > last {
			a.committed[m.Source] = m.Time
		}
	}

	a.now = t
	return due, rejected
}
