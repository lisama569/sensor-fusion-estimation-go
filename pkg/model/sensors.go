package model

import (
	"encoding/json"
	"errors"
	"math"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
)

// LinearPosition is a direct 2D position measurement [px, py] with
// covariance R. The measurement model is h(x)=H x with
//
//	H = [1 0 0 0
//	     0 1 0 0].
type LinearPosition struct{ R *mat.Matrix }

// Dim returns 2.
func (s *LinearPosition) Dim() int { return 2 }

// Measure implements MeasurementModel.
func (s *LinearPosition) Measure(x *mat.Vector) (*mat.Vector, *mat.Matrix, error) {
	if x == nil || x.Rows != 4 {
		return nil, nil, errors.New("linearposition: state must be a 4-vector")
	}
	H := mat.From([][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
	})
	return mat.MulVec(H, x), H, nil
}

// Noise implements MeasurementModel.
func (s *LinearPosition) Noise() *mat.Matrix { return s.R }

// LinearPositionKind is the stable archive kind of LinearPosition.
const LinearPositionKind = "linear_position/v1"

type linearPositionParams struct {
	R [][]float64 `json:"r"`
}

// Kind implements Serializable.
func (s *LinearPosition) Kind() string { return LinearPositionKind }

// Params implements Serializable.
func (s *LinearPosition) Params() ([]byte, error) {
	if s.R == nil {
		return nil, errors.New("linearposition: nil noise covariance")
	}
	rows := make([][]float64, s.R.Rows)
	for i := range rows {
		rows[i] = s.R.Slice(i)
	}
	return jsonParams(linearPositionParams{R: rows})
}

func decodeLinearPosition(raw []byte) (any, error) {
	var p linearPositionParams
	if err := json.Unmarshal(raw, &p); err != nil {
		return nil, err
	}
	return &LinearPosition{R: mat.From(p.R)}, nil
}

// RangeBearing is a nonlinear range/bearing measurement relative to a sensor
// placed at (Sx, Sy):
//
//	h(x) = [ sqrt(dx^2+dy^2), atan2(dy, dx) ], dx=px-Sx, dy=py-Sy
type RangeBearing struct {
	Sx, Sy float64
	R      *mat.Matrix
}

// Dim returns 2.
func (s *RangeBearing) Dim() int { return 2 }

// Measure implements MeasurementModel. The Jacobian is evaluated at x; the
// bearing is kept in the standard (-pi, pi] branch provided by atan2.
func (s *RangeBearing) Measure(x *mat.Vector) (*mat.Vector, *mat.Matrix, error) {
	if x == nil || x.Rows != 4 {
		return nil, nil, errors.New("rangebearing: state must be a 4-vector")
	}
	px, py := mat.VecAt(x, 0), mat.VecAt(x, 1)
	dx, dy := px-s.Sx, py-s.Sy
	r2 := dx*dx + dy*dy
	if r2 <= 1e-12 || math.IsNaN(r2) || math.IsInf(r2, 0) {
		return nil, nil, errors.New("rangebearing: degenerate geometry (range collapses to zero)")
	}
	r := math.Sqrt(r2)
	hx := mat.Vec(r, math.Atan2(dy, dx))
	H := mat.From([][]float64{
		{dx / r, dy / r, 0, 0},
		{-dy / r2, dx / r2, 0, 0},
	})
	return hx, H, nil
}

// Noise implements MeasurementModel.
func (s *RangeBearing) Noise() *mat.Matrix { return s.R }

// RangeBearingKind is the stable archive kind of RangeBearing.
const RangeBearingKind = "range_bearing/v1"

type rangeBearingParams struct {
	Sx float64     `json:"sx"`
	Sy float64     `json:"sy"`
	R  [][]float64 `json:"r"`
}

// Kind implements Serializable.
func (s *RangeBearing) Kind() string { return RangeBearingKind }

// Params implements Serializable.
func (s *RangeBearing) Params() ([]byte, error) {
	if s.R == nil {
		return nil, errors.New("rangebearing: nil noise covariance")
	}
	rows := make([][]float64, s.R.Rows)
	for i := range rows {
		rows[i] = s.R.Slice(i)
	}
	return jsonParams(rangeBearingParams{Sx: s.Sx, Sy: s.Sy, R: rows})
}

func decodeRangeBearing(raw []byte) (any, error) {
	var p rangeBearingParams
	if err := json.Unmarshal(raw, &p); err != nil {
		return nil, err
	}
	return &RangeBearing{Sx: p.Sx, Sy: p.Sy, R: mat.From(p.R)}, nil
}

func init() {
	RegisterMeasurementDecoder(LinearPositionKind, decodeLinearPosition)
	RegisterMeasurementDecoder(RangeBearingKind, decodeRangeBearing)
}

var (
	_ MeasurementModel = (*LinearPosition)(nil)
	_ MeasurementModel = (*RangeBearing)(nil)
)
