package engine

// HealthConfig tunes NIS gating and fault/recovery counters.
type HealthConfig struct {
	// GateSigma is the innovation gate: a NIS above GateSigma^2 counts as a
	// soft failure (implausible innovation given predicted covariance).
	GateSigma float64
	// FaultStreak consecutive soft failures (or one hard failure) isolate a
	// stream.
	FaultStreak int
	// RecoverStreak consecutive structurally-valid, in-gate trials while
	// isolated reinstate a stream. While isolated the measurements are
	// evaluated against the model for gating but are NOT applied to the
	// global estimate until recovery completes.
	RecoverStreak int
}

// DefaultHealthConfig returns conservative defaults.
func DefaultHealthConfig() HealthConfig {
	return HealthConfig{GateSigma: 5, FaultStreak: 3, RecoverStreak: 3}
}

// streamHealth tracks the fault state machine of one sensor stream.
type streamHealth struct {
	faulty     bool
	badStreak  int
	goodStreak int
}

// healthTracker holds per-stream state machines.
type healthTracker struct {
	cfg     HealthConfig
	streams map[string]*streamHealth
}

func newHealthTracker(cfg HealthConfig) *healthTracker {
	return &healthTracker{cfg: cfg, streams: make(map[string]*streamHealth)}
}

func (h *healthTracker) get(source string) *streamHealth {
	s, ok := h.streams[source]
	if !ok {
		s = &streamHealth{}
		h.streams[source] = s
	}
	return s
}

// healthy reports whether a stream is currently allowed to contribute.
func (h *healthTracker) healthy(source string) bool {
	return !h.get(source).faulty
}

// faulty reports whether a stream is isolated.
func (h *healthTracker) faulty(source string) bool { return h.get(source).faulty }

// markHardFault force-isolates a stream immediately (NaN/Inf or divergent
// update). Recovery counters are reset.
func (h *healthTracker) markHardFault(source string) {
	s := h.get(source)
	s.faulty = true
	s.badStreak = 0
	s.goodStreak = 0
}

// Observe classifies one completed in-EKF update of a healthy stream:
// hardFault isolates immediately; softFault (NIS out of gate) accumulates and
// isolates after FaultStreak. Returns isolated=true if the stream transitioned
// into isolation on this call.
func (h *healthTracker) Observe(source string, hardFault, softFault bool) (isolated bool) {
	s := h.get(source)
	if s.faulty {
		return false
	}
	if hardFault {
		s.faulty = true
		s.badStreak = 0
		s.goodStreak = 0
		return true
	}
	if softFault {
		s.badStreak++
		s.goodStreak = 0
		if s.badStreak >= h.cfg.FaultStreak {
			s.faulty = true
			s.badStreak = 0
			return true
		}
		return false
	}
	s.badStreak = 0
	return false
}

// Probe records one recovery trial from an isolated stream. A good (valid,
// in-gate) trial advances the recovery streak and reinstates the stream after
// RecoverStreak; a bad trial resets the streak. Returns recovered=true when
// the stream becomes healthy again.
func (h *healthTracker) Probe(source string, good bool) (recovered bool) {
	s := h.get(source)
	if !s.faulty {
		return false
	}
	if !good {
		s.goodStreak = 0
		return false
	}
	s.goodStreak++
	if s.goodStreak >= h.cfg.RecoverStreak {
		s.faulty = false
		s.goodStreak = 0
		s.badStreak = 0
		return true
	}
	return false
}
