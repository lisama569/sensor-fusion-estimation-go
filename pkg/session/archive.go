package session

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/lisama569/sensor-fusion-estimation-go/pkg/engine"
)

// ArchiveFormatVersion is the current self-describing archive envelope format.
const ArchiveFormatVersion = 1

// archiveMagic identifies a session archive byte image.
const archiveMagic = "SFE-SESSION"

// ErrArchiveVersion is returned when an archive advertises an unsupported
// envelope or engine snapshot version. The archive is refused outright: no
// half-loaded state is ever used for further estimation.
var ErrArchiveVersion = errors.New("session: archive version mismatch")

// ErrArchiveCorrupt is returned when an archive cannot be parsed, fails its
// checksum, has a wrong magic, or is internally inconsistent (truncated,
// tampered with, or written halfway).
var ErrArchiveCorrupt = errors.New("session: archive corrupt or truncated")

// envelope is the outer self-describing, checksummed container. The Body is
// kept as raw JSON so the checksum covers the exact serialized bytes.
type envelope struct {
	Magic   string          `json:"magic"`
	Version int             `json:"version"`
	Created string          `json:"created"`
	Algo    string          `json:"checksum_algo"`
	SHA256  string          `json:"sha256"`
	Body    json.RawMessage `json:"body"`
}

// snapshotDTO is the serialized image of a session at one point in time.
type snapshotDTO struct {
	Engine *engine.Snapshot `json:"engine"`
	Memory *memoryDTO       `json:"memory"`
	// Inflight counts submitted-but-not-yet-advanced (source,timestamp) keys,
	// so an archive boundary cannot turn a second same-batch copy into a
	// "fresh" measurement. Encoded as parallel slices for stable JSON.
	InflightSources []string  `json:"inflight_sources"`
	InflightTimes   []float64 `json:"inflight_times"`
	InflightCounts  []int     `json:"inflight_counts"`
	Horizon         float64   `json:"horizon"`
}

// exportLocked builds a checksummed byte image. The caller must hold s.mu; the
// image therefore represents one consistent point in time even while other
// goroutines continue submitting.
func (s *Session) exportLocked() ([]byte, error) {
	snap, err := s.eng.Snapshot()
	if err != nil {
		return nil, fmt.Errorf("session: snapshot engine: %w", err)
	}
	dto := &snapshotDTO{Engine: snap, Memory: s.mem.toDTO(), Horizon: s.horizon}
	// Deterministic inflight ordering (by source then time).
	type kv struct {
		k inflightKey
		n int
	}
	kvs := make([]kv, 0, len(s.inflight))
	for k, n := range s.inflight {
		kvs = append(kvs, kv{k, n})
	}
	sort.Slice(kvs, func(i, j int) bool {
		if kvs[i].k.source != kvs[j].k.source {
			return kvs[i].k.source < kvs[j].k.source
		}
		return kvs[i].k.time < kvs[j].k.time
	})
	for _, e := range kvs {
		dto.InflightSources = append(dto.InflightSources, e.k.source)
		dto.InflightTimes = append(dto.InflightTimes, e.k.time)
		dto.InflightCounts = append(dto.InflightCounts, e.n)
	}
	// Compact body MUST be checksummed over the exact bytes that survive an
	// envelope marshal/unmarshal round-trip: encoding/json compacts any
	// embedded json.RawMessage. We therefore compact explicitly and build a
	// small indented envelope around it for human readability.
	body, err := marshalCompact(dto)
	if err != nil {
		return nil, fmt.Errorf("session: encode body: %w", err)
	}
	sum := sha256.Sum256(body)
	// Pretty envelope with the raw compact body embedded.
	out := fmt.Sprintf("{\n  %q: %q,\n  %q: %d,\n  %q: %q,\n  %q: %q,\n  %q: %q,\n  %q: %s\n}\n",
		"magic", archiveMagic,
		"version", ArchiveFormatVersion,
		"created", time.Now().UTC().Format(time.RFC3339Nano),
		"checksum_algo", "sha256",
		"sha256", hex.EncodeToString(sum[:]),
		"body", string(body),
	)
	return []byte(out), nil
}

// marshalCompact returns canonical compact JSON for v.
func marshalCompact(v any) ([]byte, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// Export serializes the session to a self-describing, checksummed byte image.
func (s *Session) Export() ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.exportLocked()
}

// Import parses and verifies a byte image produced by Export. Version mismatch
// maps to ErrArchiveVersion; parse/checksum/consistency failures map to
// ErrArchiveCorrupt. No engine state is constructed until the image is fully
// validated, so a bad archive can never leave a half-initialized session.
func Import(data []byte) (*Session, error) {
	var env envelope
	if err := json.Unmarshal(data, &env); err != nil {
		return nil, fmt.Errorf("%w: cannot parse envelope: %v", ErrArchiveCorrupt, err)
	}
	if env.Magic != archiveMagic {
		return nil, fmt.Errorf("%w: bad magic %q", ErrArchiveCorrupt, env.Magic)
	}
	if env.Version != ArchiveFormatVersion {
		return nil, fmt.Errorf("%w: envelope version=%d supported=%d",
			ErrArchiveVersion, env.Version, ArchiveFormatVersion)
	}
	if env.Algo != "sha256" {
		return nil, fmt.Errorf("%w: unsupported checksum algo %q", ErrArchiveVersion, env.Algo)
	}
	if len(env.Body) == 0 {
		return nil, fmt.Errorf("%w: empty body", ErrArchiveCorrupt)
	}
	sum := sha256.Sum256(env.Body)
	if got := hex.EncodeToString(sum[:]); got != env.SHA256 {
		return nil, fmt.Errorf("%w: checksum mismatch body=%s archive=%s",
			ErrArchiveCorrupt, got, env.SHA256)
	}

	var dto snapshotDTO
	if err := json.Unmarshal(env.Body, &dto); err != nil {
		return nil, fmt.Errorf("%w: cannot parse body: %v", ErrArchiveCorrupt, err)
	}
	if dto.Engine == nil || dto.Memory == nil {
		return nil, fmt.Errorf("%w: body missing engine or memory", ErrArchiveCorrupt)
	}
	if dto.Engine.FormatVersion != engine.SnapshotFormatVersion {
		return nil, fmt.Errorf("%w: engine snapshot version=%d supported=%d",
			ErrArchiveVersion, dto.Engine.FormatVersion, engine.SnapshotFormatVersion)
	}
	mem, err := memoryFromDTO(dto.Memory)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrArchiveCorrupt, err)
	}
	eng, err := engine.NewFromSnapshot(dto.Engine)
	if err != nil {
		// Unknown model kinds / layout incompatibilities are version problems;
		// everything else is corruption. Distinguish best-effort by message.
		if errors.Is(err, engine.ErrSnapshotVersion) {
			return nil, fmt.Errorf("%w: %v", ErrArchiveVersion, err)
		}
		return nil, fmt.Errorf("%w: %v", ErrArchiveCorrupt, err)
	}
	horizon := dto.Horizon
	if horizon < 0 || horizon != dto.Engine.Config.MaxDelayHorizon {
		// Trust the engine config as the source of truth.
		horizon = dto.Engine.Config.MaxDelayHorizon
	}
	sess := &Session{eng: eng, mem: mem, horizon: horizon, inflight: make(map[inflightKey]int)}
	nInflight := len(dto.InflightSources)
	if len(dto.InflightTimes) != nInflight || len(dto.InflightCounts) != nInflight {
		return nil, fmt.Errorf("%w: inflight arrays length mismatch", ErrArchiveCorrupt)
	}
	for i := 0; i < nInflight; i++ {
		if dto.InflightCounts[i] <= 0 {
			return nil, fmt.Errorf("%w: non-positive inflight count", ErrArchiveCorrupt)
		}
		sess.inflight[inflightKey{source: dto.InflightSources[i], time: dto.InflightTimes[i]}] = dto.InflightCounts[i]
	}
	return sess, nil
}

// SaveFile atomically writes the exported image to path (0600). It writes a
// sibling temp file and renames it into place, so a crash mid-write cannot
// leave a half-written archive at path.
func (s *Session) SaveFile(path string) error {
	data, err := s.Export()
	if err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if dir == "" {
		dir = "."
	}
	tmp, err := os.CreateTemp(dir, ".session-*.tmp")
	if err != nil {
		return fmt.Errorf("session: create temp archive: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("session: write archive: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return fmt.Errorf("session: sync archive: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("session: close archive: %w", err)
	}
	if err := os.Chmod(tmpName, 0o600); err != nil {
		return fmt.Errorf("session: chmod archive: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("session: install archive: %w", err)
	}
	return nil
}

// LoadFile reads an archive from path and restores it, applying the same
// version/corruption checks as Import.
func LoadFile(path string) (*Session, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("session: read archive %q: %w", path, err)
	}
	return Import(data)
}
