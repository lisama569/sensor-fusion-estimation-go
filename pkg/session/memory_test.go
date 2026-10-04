package session

import "testing"

func TestMemoryDuplicateAndHorizonBound(t *testing.T) {
	m := newMemory(2.0, 100)
	for i := 0; i < 5; i++ {
		m.add("gps", float64(i))
	}
	if !m.seen("gps", 3) {
		t.Fatal("processed key should be seen")
	}
	if m.seen("gps", 9) {
		t.Fatal("unseen key reported as seen")
	}
	// Second add must not inflate the key count.
	m.add("gps", 3)
	if got := m.keys(); got != 5 {
		t.Fatalf("keys=%d, want 5 after duplicate add", got)
	}
	// After advancing to t=4, keys <= 2 (4-horizon) leave the explicit map:
	// they are covered by the monotonic watermark instead, so the map size
	// stays bounded while repeat protection is preserved.
	m.advance(4)
	if got := m.keys(); got != 2 {
		t.Fatalf("keys after prune=%d, want 2", got)
	}
	if !m.seen("gps", 2.0) {
		t.Fatal("horizon-pruned key must remain seen via watermark (never fused twice)")
	}
	t.Logf("horizon pruning leaves %d retained keys (bounded by horizon window)", m.keys())
}

func TestMemoryHardCapWatermark(t *testing.T) {
	const cap = 4
	m := newMemory(1_000_000, cap) // huge horizon: only the hard cap bounds it
	for i := 0; i < 10; i++ {
		m.add("imu", float64(i))
	}
	m.advance(1000)
	if got := m.keys(); got > cap {
		t.Fatalf("keys=%d exceeds hard cap %d", got, cap)
	}
	// Old keys folded into the watermark are still "seen": never fused twice.
	if !m.seen("imu", 0) || !m.seen("imu", 5) {
		t.Fatal("cap-evicted keys must remain detectable as seen via watermark")
	}
	// Round-trip through the DTO.
	d := m.toDTO()
	m2, err := memoryFromDTO(d)
	if err != nil {
		t.Fatalf("memoryFromDTO: %v", err)
	}
	for i := 0; i < 10; i++ {
		if !m2.seen("imu", float64(i)) {
			t.Fatalf("restored memory lost key %d", i)
		}
	}
	t.Logf("hard cap=%d enforced; watermark covers evicted keys; DTO round-trip OK", cap)
}
