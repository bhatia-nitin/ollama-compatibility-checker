package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"flag"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"
)

var update = flag.Bool("update", false, "rewrite testdata/header.gguf from buildFixture")

// ggufWriter emits little-endian GGUF metadata for test fixtures.
type ggufWriter struct{ buf bytes.Buffer }

func (w *ggufWriter) u32(v uint32) { _ = binary.Write(&w.buf, binary.LittleEndian, v) }
func (w *ggufWriter) u64(v uint64) { _ = binary.Write(&w.buf, binary.LittleEndian, v) }
func (w *ggufWriter) str(s string) { w.u64(uint64(len(s))); w.buf.WriteString(s) }

func (w *ggufWriter) kvU32(k string, v uint32) { w.str(k); w.u32(ggufUint32); w.u32(v) }
func (w *ggufWriter) kvStr(k, v string)        { w.str(k); w.u32(ggufString); w.str(v) }

// buildFixture returns a small GGUF header with the keys gen-catalog reads,
// interleaved with array, bool, float and 64-bit values the parser must skip.
func buildFixture(version uint32) []byte {
	w := &ggufWriter{}
	w.buf.WriteString("GGUF")
	w.u32(version)
	w.u64(0)  // tensor count
	w.u64(16) // kv count
	w.kvStr("general.architecture", "llama")
	w.kvStr("general.name", "Fixture")
	w.kvU32("general.file_type", 15)
	// string array, like tokenizer.ggml.tokens
	w.str("tokenizer.ggml.tokens")
	w.u32(ggufArray)
	w.u32(ggufString)
	w.u64(3)
	for _, s := range []string{"<s>", "</s>", "hello"} {
		w.str(s)
	}
	// float32 array, like tokenizer.ggml.scores
	w.str("tokenizer.ggml.scores")
	w.u32(ggufArray)
	w.u32(ggufFloat32)
	w.u64(3)
	for i := 0; i < 3; i++ {
		w.u32(math.Float32bits(float32(i)))
	}
	w.str("general.flag")
	w.u32(ggufBool)
	w.buf.WriteByte(1)
	w.str("general.big")
	w.u32(ggufUint64)
	w.u64(1 << 40)
	w.str("general.ratio")
	w.u32(ggufFloat64)
	w.u64(math.Float64bits(0.5))
	w.kvU32("llama.block_count", 22)
	w.kvU32("llama.context_length", 2048)
	w.kvU32("llama.embedding_length", 2048)
	w.kvU32("llama.attention.head_count", 32)
	w.kvU32("llama.attention.head_count_kv", 4)
	w.kvU32("llama.attention.key_length", 128)
	w.kvU32("llama.expert_count", 8)
	w.kvU32("llama.expert_used_count", 2)
	return w.buf.Bytes()
}

func TestParseGGUFHeader(t *testing.T) {
	path := filepath.Join("testdata", "header.gguf")
	want := buildFixture(3)
	if *update {
		if err := os.WriteFile(path, want, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, want) {
		t.Fatal("testdata/header.gguf is stale; run go test ./scripts/gen-catalog -run TestParseGGUFHeader -update")
	}

	meta, err := parseGGUF(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if meta.Version != 3 {
		t.Errorf("version = %d, want 3", meta.Version)
	}
	checks := map[string]uint64{
		"llama.block_count":             22,
		"llama.attention.head_count_kv": 4,
		"llama.attention.key_length":    128,
		"llama.expert_count":            8,
		"llama.expert_used_count":       2,
		"general.file_type":             15,
		"general.big":                   1 << 40,
	}
	for k, v := range checks {
		got, ok := meta.Uint(k)
		if !ok || got != v {
			t.Errorf("%s = %d (present %v), want %d", k, got, ok, v)
		}
	}
	if s, _ := meta.String("general.architecture"); s != "llama" {
		t.Errorf("architecture = %q, want llama", s)
	}

	info, err := archInfoFrom(meta)
	if err != nil {
		t.Fatal(err)
	}
	wantInfo := archInfo{Arch: "llama", Layers: 22, KVHeads: 4, HeadDim: 128, MaxCtx: 2048, ExpertCount: 8, ExpertUsed: 2, FileType: 15}
	if info != wantInfo {
		t.Errorf("archInfo = %+v, want %+v", info, wantInfo)
	}
}

func TestParseGGUFv2(t *testing.T) {
	meta, err := parseGGUF(bytes.NewReader(buildFixture(2)))
	if err != nil {
		t.Fatal(err)
	}
	if n, _ := meta.Uint("llama.block_count"); n != 22 {
		t.Fatalf("block_count = %d, want 22", n)
	}
}

func TestParseGGUFTruncated(t *testing.T) {
	data := buildFixture(3)
	meta, err := parseGGUF(bytes.NewReader(data[:len(data)-6]))
	if !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("err = %v, want io.ErrUnexpectedEOF", err)
	}
	if n, ok := meta.Uint("llama.block_count"); !ok || n != 22 {
		t.Fatalf("partial metadata lost: block_count = %d, %v", n, ok)
	}
}

func TestHeadDimFallback(t *testing.T) {
	meta := ggufMeta{KV: map[string]any{
		"general.architecture":          "qwen2",
		"qwen2.block_count":             uint64(28),
		"qwen2.context_length":          uint64(32768),
		"qwen2.embedding_length":        uint64(3584),
		"qwen2.attention.head_count":    uint64(28),
		"qwen2.attention.head_count_kv": uint64(4),
	}}
	info, err := archInfoFrom(meta)
	if err != nil {
		t.Fatal(err)
	}
	if info.HeadDim != 128 || info.ExpertCount != 0 {
		t.Fatalf("info = %+v, want head_dim 128 and no experts", info)
	}
}

func TestQuantFromBareTag(t *testing.T) {
	if got := quantFromTag("glm-4.7-flash:q8_0"); got != "q8_0" {
		t.Errorf("got %q", got)
	}
	if got := quantFromTag("granite4:3b-h"); got != "" {
		t.Errorf("size tag got %q", got)
	}
}

func TestQuantName(t *testing.T) {
	cases := []struct {
		tag      string
		fileType uint64
		want     string
	}{
		{"llama3.1:8b-instruct-q4_K_M", 0, "q4_K_M"},
		{"qwen2.5:7b-instruct-q8_0", 0, "q8_0"},
		{"gemma3:27b-it-q3_K_M", 0, "q3_K_M"},
		{"phi4:14b-fp16", 0, "fp16"},
		{"llama3.2:3b", 15, "q4_K_M"},
		{"llama3.2:3b", 7, "q8_0"},
		{"llama3.2:3b", 18, "q6_K"},
	}
	for _, c := range cases {
		if got := quantName(c.tag, c.fileType); got != c.want {
			t.Errorf("quantName(%q, %d) = %q, want %q", c.tag, c.fileType, got, c.want)
		}
	}
}

func TestBits(t *testing.T) {
	if got := bits(4_920_000_000, 8.0); got != 4.92 {
		t.Fatalf("bits = %v, want 4.92", got)
	}
}

func TestActiveParamsOverride(t *testing.T) {
	moe := archInfo{ExpertCount: 128, ExpertUsed: 8}
	override := 3.3
	if got := activeParams(source{ParamsB: 30.5, ActiveParamsB: &override}, moe); got != 3.3 {
		t.Errorf("override: active = %v, want 3.3", got)
	}
	if got := activeParams(source{ParamsB: 30.5}, moe); got != 1.91 {
		t.Errorf("formula fallback: active = %v, want 1.91", got)
	}
	if got := activeParams(source{ParamsB: 8}, archInfo{}); got != 8 {
		t.Errorf("dense: active = %v, want 8", got)
	}
}
