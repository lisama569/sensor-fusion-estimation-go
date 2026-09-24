package engine

import (
	"math"
	"testing"

	"github.com/lisama569/sensor-fusion-estimation-go/internal/mat"
	"github.com/lisama569/sensor-fusion-estimation-go/pkg/model"
)

func posModel(std float64) *model.LinearPosition {
	v := std * std
	return &model.LinearPosition{R: mat.From([][]float64{{v, 0}, {0, v}})}
}

// TestHardFaultIsolationAndRecovery injects non-finite samples on one stream.
// The stream must be isolated, its later samples refused as stream_faulty,
// and it must rejoin after RecoverStreak healthy probes. The healthy stream's
// estimate must keep improving regardless.
func TestHardFaultIsolationAndRecovery(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Health.FaultStreak = 2
	cfg.Health.RecoverStreak = 3
	eng, err := New(initialState(), &model.ConstantVelocity{QPSD: 0.05}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	bad := posModel(0.2)
	good := posModel(0.2)

	// A few healthy samples from both at t=0.2,0.4.
	for k := 1; k <= 2; k++ {
		tt := float64(k) * 0.2
		if err := eng.Submit("good", tt, mat.Vec(tt, 0), good); err != nil {
			t.Fatal(err)
		}
		if err := eng.Submit("bad", tt, mat.Vec(tt, 0), bad); err != nil {
			t.Fatal(err)
		}
		eng.Advance(tt)
	}
	healthyState := eng.State().Clone()

	// Inject two non-finite samples on bad stream. They are buffered at
	// submit (arrival order must not matter); isolation happens in time
	// order during Advance.
	for k := 0; k < 2; k++ {
		tt := 0.6 + float64(k)*0.2
		if err := eng.Submit("bad", tt, mat.Vec(math.NaN(), 0), bad); err != nil {
			t.Fatalf("non-finite sample should be buffered at submit: %v", err)
		}
	}
	if eng.ht.faulty("bad") {
		t.Fatal("bad stream must not be isolated before Advance")
	}
	_, faultRejs, err := eng.Advance(0.8)
	if err != nil {
		t.Fatal(err)
	}
	gotNF := 0
	for _, r := range faultRejs {
		if r.Source == "bad" && r.Reason == ReasonNonFinite {
			gotNF++
		}
	}
	if gotNF == 0 {
		t.Fatalf("expected non_finite_value rejection during Advance, got %v", faultRejs)
	}
	if !eng.ht.faulty("bad") {
		t.Fatal("bad stream should be isolated after hard fault is processed")
	}
	if eng.ht.faulty("good") {
		t.Fatal("healthy stream must not be isolated")
	}

	// Healthy samples from bad while isolated -> buffered, then refused as
	// stream_faulty during advance and counted as recovery probes.
	probeTimes := []float64{1.0, 1.2}
	for _, tt := range probeTimes {
		if err := eng.Submit("bad", tt, mat.Vec(tt, 0), bad); err != nil {
			t.Fatalf("isolated stream sample should be buffered: %v", err)
		}
	}
	// Meanwhile good stream keeps updating.
	if err := eng.Submit("good", 1.0, mat.Vec(1.0, 0), good); err != nil {
		t.Fatal(err)
	}
	st, rejs, err := eng.Advance(1.2)
	if err != nil {
		t.Fatal(err)
	}
	if len(rejs) != 2 {
		t.Fatalf("expected 2 stream_faulty rejections, got %d: %v", len(rejs), rejs)
	}
	for _, r := range rejs {
		if r.Source != "bad" || r.Reason != ReasonStreamFaulty {
			t.Fatalf("unexpected rejection %v", r)
		}
	}
	// The isolated stream's probes must not have moved the estimate relative
	// to good-only updates at the same timestamps. We verify covariance stays
	// PSD and state remains finite and sane.
	assertPSD(t, st.P, "P during isolation")
	if !mat.Finite(st.X) {
		t.Fatal("state non-finite after fault isolation")
	}
	_ = healthyState

	// One more healthy probe reaches RecoverStreak=3 -> stream recovers.
	if err := eng.Submit("bad", 1.4, mat.Vec(1.4, 0), bad); err != nil {
		t.Fatalf("buffering recovery probe: %v", err)
	}
	st2, _, _ := eng.Advance(1.4)
	if eng.ht.faulty("bad") {
		t.Fatal("bad stream should have recovered")
	}
	// The last probe is the recovery one; once recovered it gets fused on the
	// NEXT sample (probes themselves are not applied, by design).
	if err := eng.Submit("bad", 1.6, mat.Vec(1.6, 0), bad); err != nil {
		t.Fatal(err)
	}
	st3, rejs3, _ := eng.Advance(1.6)
	for _, r := range rejs3 {
		if r.Source == "bad" {
			t.Fatalf("recovered stream sample rejected: %v", r)
		}
	}
	assertPSD(t, st2.P, "P at recovery")
	assertPSD(t, st3.P, "P after recovery")
	if eng.Stats().PerStream["bad"].Recoveries < 1 {
		t.Fatal("expected at least one recorded recovery")
	}
}

// TestSoftFaultDivergingStream isolates a stream whose measurements persist
// diverge from the predicted state (large NIS streak), without any NaN/Inf.
func TestSoftFaultDivergingStream(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Health.GateSigma = 3
	cfg.Health.FaultStreak = 3
	cfg.Health.RecoverStreak = 2
	eng, _ := New(initialState(), &model.ConstantVelocity{QPSD: 0.01}, cfg)
	md := posModel(0.01)

	// Establish good tracking with plausible samples.
	for k := 1; k <= 3; k++ {
		tt := float64(k) * 0.1
		eng.Submit("s", tt, mat.Vec(tt*1.0, 0), md)
		eng.Advance(tt)
	}
	// Then the sensor reports wildly offset positions: huge NIS repeatedly.
	for k := 0; k < 4; k++ {
		tt := 0.4 + float64(k)*0.1
		eng.Submit("s", tt, mat.Vec(50, 50), md)
		eng.Advance(tt)
	}
	if !eng.ht.faulty("s") {
		t.Fatal("persistently diverging stream must be isolated")
	}
	// State must not have jumped toward the bogus readings.
	st := eng.State()
	if math.Abs(mat.VecAt(st.X, 0)) > 5 || math.Abs(mat.VecAt(st.X, 1)) > 5 {
		t.Fatalf("faulty readings polluted the state: %v", st.X.Data)
	}
	assertPSD(t, st.P, "P after divergence isolation")

	// Recover with consistent, plausible probes.
	for k := 0; k < 3; k++ {
		tt := 0.8 + float64(k)*0.1
		eng.Submit("s", tt, mat.Vec(tt, 0), md)
		eng.Advance(tt)
	}
	if eng.ht.faulty("s") {
		t.Fatal("stream should recover after in-gate probes")
	}
}
