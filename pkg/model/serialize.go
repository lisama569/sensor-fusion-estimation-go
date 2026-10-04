// Model serialization support for long-lived estimation sessions.
//
// A model that participates in session save/resume must be self-describing:
// it reports a stable Kind string and can round-trip its parameters through
// JSON. Built-in models (ConstantVelocity, LinearPosition, RangeBearing) are
// registered automatically; custom models can register their own decoders.
package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"sync"
)

// Serializable is implemented by models that can be embedded in a session
// archive.
type Serializable interface {
	// Kind is the stable, unique identifier of the model type. The built-in
	// kinds carry a "/vN" suffix so an incompatible parameter layout can be
	// introduced as a new kind instead of silently mis-decoding an old archive.
	Kind() string
	// Params encodes every parameter needed to reconstruct an equivalent
	// model as JSON.
	Params() ([]byte, error)
}

// SerializableMotion is a MotionModel that can be serialized.
type SerializableMotion interface {
	MotionModel
	Serializable
}

// SerializableMeasurement is a MeasurementModel that can be serialized.
type SerializableMeasurement interface {
	MeasurementModel
	Serializable
}

// ModelDecoder rebuilds a model from the parameters emitted by Params.
type ModelDecoder func(raw []byte) (any, error)

var (
	regMu      sync.RWMutex
	motionDec  = map[string]ModelDecoder{}
	measureDec = map[string]ModelDecoder{}
)

// ErrNotSerializable is returned when a model cannot be archived.
var ErrNotSerializable = errors.New("model: model is not serializable (register a decoder)")

// ErrUnknownKind is returned when no decoder is registered for a kind.
var ErrUnknownKind = errors.New("model: unknown model kind (decoder not registered in this process)")

// RegisterMotionDecoder registers a decoder for a motion model kind. It is
// typically called from an init function. Registering an empty kind or a nil
// decoder is a programming error and panics.
func RegisterMotionDecoder(kind string, dec ModelDecoder) {
	if kind == "" || dec == nil {
		panic("model: RegisterMotionDecoder requires kind and decoder")
	}
	regMu.Lock()
	defer regMu.Unlock()
	motionDec[kind] = dec
}

// RegisterMeasurementDecoder registers a decoder for a measurement model kind.
func RegisterMeasurementDecoder(kind string, dec ModelDecoder) {
	if kind == "" || dec == nil {
		panic("model: RegisterMeasurementDecoder requires kind and decoder")
	}
	regMu.Lock()
	defer regMu.Unlock()
	measureDec[kind] = dec
}

// EncodeMotion returns the kind and encoded parameters of a serializable
// motion model.
func EncodeMotion(m MotionModel) (string, []byte, error) {
	sm, ok := m.(SerializableMotion)
	if !ok {
		return "", nil, fmt.Errorf("%w: motion %T", ErrNotSerializable, m)
	}
	raw, err := sm.Params()
	if err != nil {
		return "", nil, fmt.Errorf("model: encode motion %s: %w", sm.Kind(), err)
	}
	return sm.Kind(), raw, nil
}

// EncodeMeasurement returns the kind and encoded parameters of a serializable
// measurement model.
func EncodeMeasurement(m MeasurementModel) (string, []byte, error) {
	sm, ok := m.(SerializableMeasurement)
	if !ok {
		return "", nil, fmt.Errorf("%w: measurement %T", ErrNotSerializable, m)
	}
	raw, err := sm.Params()
	if err != nil {
		return "", nil, fmt.Errorf("model: encode measurement %s: %w", sm.Kind(), err)
	}
	return sm.Kind(), raw, nil
}

// DecodeMotion rebuilds a motion model from kind and parameters.
func DecodeMotion(kind string, raw []byte) (MotionModel, error) {
	regMu.RLock()
	dec, ok := motionDec[kind]
	regMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownKind, kind)
	}
	v, err := dec(raw)
	if err != nil {
		return nil, fmt.Errorf("model: decode motion %q: %w", kind, err)
	}
	mm, ok := v.(MotionModel)
	if !ok {
		return nil, fmt.Errorf("model: decoder for %q produced %T, not MotionModel", kind, v)
	}
	return mm, nil
}

// DecodeMeasurement rebuilds a measurement model from kind and parameters.
func DecodeMeasurement(kind string, raw []byte) (MeasurementModel, error) {
	regMu.RLock()
	dec, ok := measureDec[kind]
	regMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrUnknownKind, kind)
	}
	v, err := dec(raw)
	if err != nil {
		return nil, fmt.Errorf("model: decode measurement %q: %w", kind, err)
	}
	mm, ok := v.(MeasurementModel)
	if !ok {
		return nil, fmt.Errorf("model: decoder for %q produced %T, not MeasurementModel", kind, v)
	}
	return mm, nil
}

// jsonParams is a small helper for the built-in models.
func jsonParams(v any) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return b, nil
}
