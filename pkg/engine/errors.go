// Package engine implements an extended Kalman filter with time alignment of
// multi-rate, delayed, out-of-order measurements and per-stream fault
// isolation. All entry points are safe for concurrent use.
package engine

import (
	"errors"
	"fmt"
)

// Sentinel errors for low-level numerical/model failures. The engine maps
// these to typed *RejectError values with a machine-readable RejectReason.
var (
	// ErrIllegalNoise: a covariance is not finite, symmetric or PSD.
	ErrIllegalNoise = errors.New("illegal noise covariance")
	// ErrDimensionMismatch: vector/matrix/model dimensions disagree.
	ErrDimensionMismatch = errors.New("dimension mismatch")
	// ErrDegenerate: the innovation covariance is singular (no information).
	ErrDegenerate = errors.New("degenerate innovation covariance")
	// ErrIllConditioned: the update problem exceeds the condition limit.
	ErrIllConditioned = errors.New("ill-conditioned update")
	// ErrNonFinite: a state or measurement contains NaN/+Inf/-Inf.
	ErrNonFinite = errors.New("non-finite value encountered")
	// ErrBadModel: the measurement model itself failed (e.g. singular point).
	ErrBadModel = errors.New("measurement model evaluation failed")
)

// RejectReason is the machine-readable category of a rejected measurement.
type RejectReason string

const (
	// ReasonNonFinite: a measurement contains NaN/+Inf/-Inf.
	ReasonNonFinite RejectReason = "non_finite_value"
	// ReasonDuplicate: same source already submitted a measurement at this
	// timestamp.
	ReasonDuplicate RejectReason = "duplicate_timestamp"
	// ReasonExpired: measurement timestamp is older than the last committed
	// time beyond the allowed delay horizon.
	ReasonExpired RejectReason = "expired_measurement"
	// ReasonBadTimestamp: time is not finite.
	ReasonBadTimestamp RejectReason = "bad_timestamp"
	// ReasonDimensionMismatch: measurement/noise/model dimensions disagree.
	ReasonDimensionMismatch RejectReason = "dimension_mismatch"
	// ReasonIllegalNoise: noise covariance is not symmetric PSD or not finite.
	ReasonIllegalNoise RejectReason = "illegal_noise"
	// ReasonDegenerate: the innovation covariance is singular (no
	// information).
	ReasonDegenerate RejectReason = "degenerate_measurement"
	// ReasonIllConditioned: the update problem is numerically too
	// ill-conditioned to proceed safely.
	ReasonIllConditioned RejectReason = "ill_conditioned"
	// ReasonStreamFaulty: the source is currently isolated due to earlier
	// faults; measurements are refused until it recovers.
	ReasonStreamFaulty RejectReason = "stream_faulty"
	// ReasonModelFailed: h(x)/H could not be evaluated (e.g. singular point).
	ReasonModelFailed RejectReason = "model_evaluation_failed"
	// ReasonOutOfGate: NIS exceeded the innovation gate; the sample is
	// withheld from the estimate (a soft reject, not an isolation by itself).
	ReasonOutOfGate RejectReason = "out_of_gate"
	// ReasonAlreadyProcessed: the exact (source, timestamp) pair was already
	// fused before an archive boundary or in an earlier offline replay batch.
	// Refused at the session layer so a measurement can never be fused twice.
	// Distinct from duplicate_timestamp (two copies within one processing
	// batch) and from expired_measurement (too old to be usable at all).
	ReasonAlreadyProcessed RejectReason = "already_processed"
)

// RejectError records why a measurement was refused instead of being fused.
type RejectError struct {
	Source string
	Time   float64
	Reason RejectReason
	Detail string
}

// Error implements error.
func (e *RejectError) Error() string {
	return fmt.Sprintf("measurement rejected: source=%q time=%.6f reason=%s detail=%s",
		e.Source, e.Time, e.Reason, e.Detail)
}

func reject(source string, t float64, reason RejectReason, detail string) *RejectError {
	return &RejectError{Source: source, Time: t, Reason: reason, Detail: detail}
}

// Reject builds a typed rejection. It is exported so layers above the engine
// (e.g. the session package) can produce rejections with the same type and
// machine-readable reason taxonomy.
func Reject(source string, t float64, reason RejectReason, detail string) *RejectError {
	return reject(source, t, reason, detail)
}
