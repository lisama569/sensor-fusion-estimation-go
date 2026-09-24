// Package reject defines distinguishable rejection reasons for invalid
// measurements, noise configurations and degenerate filter updates.
package reject

import "fmt"

// Reason is a stable, distinguishable code explaining why an input or an
// operation was rejected instead of being silently dropped.
type Reason string

const (
	// ReasonUnknown is the zero value and must never be returned by a
	// validated path.
	ReasonUnknown Reason = ""

	// ReasonNonFiniteValue: a measurement vector contains NaN/Inf.
	ReasonNonFiniteValue Reason = "non_finite_measurement"
	// ReasonBadTimestamp: timestamp is NaN.
	ReasonBadTimestamp Reason = "bad_timestamp"
	// ReasonStaleTimestamp: timestamp is older than the already committed
	// watermark (out of the tolerated delay window).
	ReasonStaleTimestamp Reason = "stale_timestamp"
	// ReasonDuplicateTimestamp: the same sensor already submitted this
	// exact timestamp.
	ReasonDuplicateTimestamp Reason = "duplicate_timestamp"
	// ReasonNoiseNotSPD: a noise/process-covariance matrix failed the
	// symmetric positive-(semi)definite validation.
	ReasonNoiseNotSPD Reason = "noise_not_spd"
	// ReasonDimensionMismatch: vector/matrix dimensions are inconsistent.
	ReasonDimensionMismatch Reason = "dimension_mismatch"
	// ReasonDegenerateUpdate: S = H P H' + R is singular to working
	// precision, the Kalman gain cannot be formed.
	ReasonDegenerateUpdate Reason = "degenerate_update"
	// ReasonDiverging: the channel's normalized innovation squared (NIS)
	// exceeded its threshold for too long; the channel is isolated.
	ReasonDiverging Reason = "channel_diverging"
	// ReasonChannelIsolated: measurement arrived while its channel is
	// fault-isolated.
	ReasonChannelIsolated Reason = "channel_isolated"
	// ReasonNonFiniteInnovation: the predicted measurement or innovation
	// is not finite (nonlinear evaluation blew up).
	ReasonNonFiniteInnovation Reason = "non_finite_innovation"
	// ReasonNotInitialized: an update was requested before the first valid
	// state estimate existed.
	ReasonNotInitialized Reason = "not_initialized"
	// ReasonNoData: drain/flush was called with an empty alignment buffer.
	ReasonNoData Reason = "no_data"
)

// Error wraps a rejection reason with a human readable detail.
type Error struct {
	Reason Reason
	Detail string
}

func (e *Error) Error() string {
	if e.Detail == "" {
		return string(e.Reason)
	}
	return fmt.Sprintf("%s: %s", e.Reason, e.Detail)
}

// New builds a *Error with the given reason and detail.
func New(r Reason, format string, args ...any) *Error {
	return &Error{Reason: r, Detail: fmt.Sprintf(format, args...)}
}

// AsError extracts *Error from err and reports whether err is a rejection.
func AsError(err error) (*Error, bool) {
	e, ok := err.(*Error)
	return e, ok
}

// Of returns the rejection reason of err, or ReasonUnknown if err is nil or
// not a rejection.
func Of(err error) Reason {
	if e, ok := AsError(err); ok {
		return e.Reason
	}
	return ReasonUnknown
}
