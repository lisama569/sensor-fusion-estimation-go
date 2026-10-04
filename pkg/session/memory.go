// Package session adds long-lived, resumable estimation sessions on top of
// pkg/engine: bounded processed-measurement memory, historical replay
// (backfill), and versioned, checksummed save/restore.
package session

import (
	"errors"
	"fmt"
	"sort"

	"github.com/lisama569/sensor-fusion-estimation-go/pkg/engine"
)

const (
	// ReasonAlreadyProcessed: the exact (source, timestamp) pair was already
	// recorded as processed, either before an archive boundary or in an
	// earlier replay batch. It must never be fused twice, so it is refused
	// without touching the estimate. Distinguishable from the engine-level
	// duplicate_timestamp (two copies in the same batch) and from
	// expired_measurement (too old to be usable at all).
	ReasonAlreadyProcessed = engine.ReasonAlreadyProcessed
)

// DefaultMaxProcessedPerSource is the hard cap on retained processed keys per
// source. Together with delay-horizon pruning it gives the memory an explicit
// upper bound regardless of how long the session runs.
const DefaultMaxProcessedPerSource = 4096

// memory is the BOUNDED record of processed measurements. It remembers exactly
// which (source, timestamp) pairs were already fused so a measurement can never
// be fused twice across archives or replay batches.
//
// Bounding strategy (two layers):
//  1. Horizon pruning: after every advance, keys at or below (now - horizon)
//     are dropped. Such timestamps can never legally arrive again — any repeat
//     would be rejected by the engine as expired_measurement anyway.
//  2. Hard per-source cap: if a source still holds more than maxPerSource
//     keys, the smallest timestamps are folded into a monotonic watermark so
//     they remain "seen" (no double fusion) but stop occupying map space. A
//     genuinely-new-but-very-old sample arriving beyond the cap is then
//     refused as already_processed; this trade-off only activates under
//     sustained input rates far above the horizon/key budget and keeps the
//     memory strictly bounded.
type memory struct {
	horizon   float64
	maxPerSrc int
	sources   map[string]*srcMemory
	watermark map[string]float64
	totalKeys int
}

type srcMemory struct {
	keys map[float64]struct{}
}

// newMemory constructs a bounded processed-memory tracker. horizon must be
// finite and >= 0; maxPerSource <= 0 selects DefaultMaxProcessedPerSource.
func newMemory(horizon float64, maxPerSource int) *memory {
	if maxPerSource <= 0 {
		maxPerSource = DefaultMaxProcessedPerSource
	}
	return &memory{
		horizon:   horizon,
		maxPerSrc: maxPerSource,
		sources:   make(map[string]*srcMemory),
		watermark: make(map[string]float64),
	}
}

func (m *memory) src(source string) *srcMemory {
	s, ok := m.sources[source]
	if !ok {
		s = &srcMemory{keys: make(map[float64]struct{})}
		m.sources[source] = s
	}
	return s
}

// seen reports whether (source,t) is recorded: covered by the source
// watermark, or present as an explicit key.
func (m *memory) seen(source string, t float64) bool {
	if w, ok := m.watermark[source]; ok && t <= w {
		return true
	}
	s, ok := m.sources[source]
	if !ok {
		return false
	}
	_, dup := s.keys[t]
	return dup
}

// add records (source,t). Duplicates are not double-counted.
func (m *memory) add(source string, t float64) {
	if m.seen(source, t) {
		return
	}
	s := m.src(source)
	s.keys[t] = struct{}{}
	m.totalKeys++
}

// forget removes an explicit key. It is used when a key was optimistically
// recorded at submission but the engine later refused that measurement (e.g.
// both copies of a same-batch duplicate group), so a future genuine copy is not
// wrongly blocked. Watermark-covered keys cannot be forgotten: those are known
// processed for good.
func (m *memory) forget(source string, t float64) {
	if w, ok := m.watermark[source]; ok && t <= w {
		return
	}
	if s, ok := m.sources[source]; ok {
		if _, exists := s.keys[t]; exists {
			delete(s.keys, t)
			m.totalKeys--
		}
	}
}

// pruneHorizon drops keys at or below cutoff (they could only return as
// expired measurements now). Watermarks are advanced at least to cutoff.
func (m *memory) pruneHorizon(cutoff float64) {
	for source, s := range m.sources {
		for t := range s.keys {
			if t <= cutoff {
				delete(s.keys, t)
				m.totalKeys--
			}
		}
		if len(s.keys) == 0 {
			delete(m.sources, source)
		}
		if w := m.watermark[source]; w < cutoff {
			m.watermark[source] = cutoff
		}
	}
}

// enforceCap folds the oldest keys of any over-budget source into its
// watermark until it fits the hard per-source cap.
func (m *memory) enforceCap() {
	for source, s := range m.sources {
		if len(s.keys) <= m.maxPerSrc {
			continue
		}
		ts := make([]float64, 0, len(s.keys))
		for t := range s.keys {
			ts = append(ts, t)
		}
		sort.Float64s(ts)
		drop := len(ts) - m.maxPerSrc
		for i := 0; i < drop; i++ {
			delete(s.keys, ts[i])
			m.totalKeys--
		}
		// Everything up to and including the last dropped key is now "seen".
		if w := m.watermark[source]; w < ts[drop-1] {
			m.watermark[source] = ts[drop-1]
		}
	}
}

// advance applies horizon pruning and the hard cap. now is the latest
// committed engine time.
func (m *memory) advance(now float64) {
	m.pruneHorizon(now - m.horizon)
	m.enforceCap()
}

// keys returns the number of explicitly retained keys across all sources.
func (m *memory) keys() int { return m.totalKeys }

// boundedSources reports the current number of distinct sources (for tests and
// diagnostics).
func (m *memory) boundedSources() int { return len(m.sources) }

// memoryDTO is the serialized form of memory.
type memoryDTO struct {
	Horizon   float64
	MaxPerSrc int
	// Keys lists every explicit retained timestamp per source (sorted on
	// output so archives of the same logical state are byte-stable).
	Keys      map[string][]float64
	Watermark map[string]float64
}

func (m *memory) toDTO() *memoryDTO {
	d := &memoryDTO{
		Horizon:   m.horizon,
		MaxPerSrc: m.maxPerSrc,
		Keys:      make(map[string][]float64, len(m.sources)),
		Watermark: make(map[string]float64, len(m.watermark)),
	}
	for source, s := range m.sources {
		if len(s.keys) == 0 {
			continue
		}
		ts := make([]float64, 0, len(s.keys))
		for t := range s.keys {
			ts = append(ts, t)
		}
		sort.Float64s(ts)
		d.Keys[source] = ts
	}
	for source, w := range m.watermark {
		d.Watermark[source] = w
	}
	return d
}

func memoryFromDTO(d *memoryDTO) (*memory, error) {
	if d == nil {
		return nil, errors.New("session: nil memory snapshot")
	}
	if d.Horizon < 0 {
		return nil, fmt.Errorf("session: memory horizon %v invalid", d.Horizon)
	}
	if d.MaxPerSrc < 0 {
		return nil, fmt.Errorf("session: memory cap %d invalid", d.MaxPerSrc)
	}
	m := newMemory(d.Horizon, d.MaxPerSrc)
	for source, ts := range d.Keys {
		s := m.src(source)
		prev := -1.0
		havePrev := false
		for _, t := range ts {
			if havePrev && t < prev {
				return nil, errors.New("session: memory keys not sorted")
			}
			prev, havePrev = t, true
			if _, dup := s.keys[t]; dup {
				return nil, fmt.Errorf("session: duplicate memory key %s@%.6f", source, t)
			}
			s.keys[t] = struct{}{}
			m.totalKeys++
		}
	}
	for source, w := range d.Watermark {
		m.watermark[source] = w
	}
	// Re-apply the bound defensively in case the archive was hand-edited.
	m.enforceCap()
	return m, nil
}
