package main

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// GGUF metadata value types (gguf spec, v2 and v3).
const (
	ggufUint8   = 0
	ggufInt8    = 1
	ggufUint16  = 2
	ggufInt16   = 3
	ggufUint32  = 4
	ggufInt32   = 5
	ggufFloat32 = 6
	ggufBool    = 7
	ggufString  = 8
	ggufArray   = 9
	ggufUint64  = 10
	ggufInt64   = 11
	ggufFloat64 = 12
)

// maxStringLen guards against garbage lengths in a corrupt header.
const maxStringLen = 1 << 24

// ggufArrayInfo records an array value whose contents were skipped.
type ggufArrayInfo struct {
	ElemType uint32
	Len      uint64
}

// maxKeptArray is the longest integer or bool array kept in ggufMeta; longer
// arrays (vocabularies, merges) are skipped. Per-layer arrays are short.
const maxKeptArray = 4096

// ggufMeta is the decoded key/value header. Unsigned integers are stored as
// uint64, signed as int64, floats as float64. Integer and bool arrays up to
// maxKeptArray elements are kept as []int64 (bool as 0 or 1); other arrays
// are skipped and recorded as ggufArrayInfo.
type ggufMeta struct {
	Version uint32
	KV      map[string]any
}

// Uint returns an integer value as uint64 (negative values report false).
func (m ggufMeta) Uint(key string) (uint64, bool) {
	switch v := m.KV[key].(type) {
	case uint64:
		return v, true
	case int64:
		if v >= 0 {
			return uint64(v), true
		}
	}
	return 0, false
}

// Ints returns an integer or bool array value.
func (m ggufMeta) Ints(key string) ([]int64, bool) {
	v, ok := m.KV[key].([]int64)
	return v, ok
}

// String returns a string value.
func (m ggufMeta) String(key string) (string, bool) {
	s, ok := m.KV[key].(string)
	return s, ok
}

type ggufReader struct {
	r *bufio.Reader
}

func (g ggufReader) u32() (uint32, error) {
	var b [4]byte
	if _, err := io.ReadFull(g.r, b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b[:]), nil
}

func (g ggufReader) u64() (uint64, error) {
	var b [8]byte
	if _, err := io.ReadFull(g.r, b[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b[:]), nil
}

func (g ggufReader) str() (string, error) {
	n, err := g.u64()
	if err != nil {
		return "", err
	}
	if n > maxStringLen {
		return "", fmt.Errorf("gguf: string length %d too large", n)
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(g.r, b); err != nil {
		return "", err
	}
	return string(b), nil
}

func (g ggufReader) skip(n uint64) error {
	for n > 0 {
		chunk := n
		if chunk > math.MaxInt32 {
			chunk = math.MaxInt32
		}
		d, err := g.r.Discard(int(chunk))
		n -= uint64(d)
		if err != nil {
			return err
		}
	}
	return nil
}

// fixedSize is the byte width of scalar types, 0 for string and array.
func fixedSize(t uint32) (uint64, error) {
	switch t {
	case ggufUint8, ggufInt8, ggufBool:
		return 1, nil
	case ggufUint16, ggufInt16:
		return 2, nil
	case ggufUint32, ggufInt32, ggufFloat32:
		return 4, nil
	case ggufUint64, ggufInt64, ggufFloat64:
		return 8, nil
	case ggufString, ggufArray:
		return 0, nil
	}
	return 0, fmt.Errorf("gguf: unknown value type %d", t)
}

func (g ggufReader) scalar(t uint32) (any, error) {
	size, err := fixedSize(t)
	if err != nil {
		return nil, err
	}
	if t == ggufString {
		return g.str()
	}
	var b [8]byte
	if _, err := io.ReadFull(g.r, b[:size]); err != nil {
		return nil, err
	}
	le := binary.LittleEndian
	switch t {
	case ggufUint8:
		return uint64(b[0]), nil
	case ggufInt8:
		return int64(int8(b[0])), nil
	case ggufUint16:
		return uint64(le.Uint16(b[:])), nil
	case ggufInt16:
		return int64(int16(le.Uint16(b[:]))), nil
	case ggufUint32:
		return uint64(le.Uint32(b[:])), nil
	case ggufInt32:
		return int64(int32(le.Uint32(b[:]))), nil
	case ggufFloat32:
		return float64(math.Float32frombits(le.Uint32(b[:]))), nil
	case ggufBool:
		return b[0] != 0, nil
	case ggufUint64:
		return le.Uint64(b[:]), nil
	case ggufInt64:
		return int64(le.Uint64(b[:])), nil
	case ggufFloat64:
		return math.Float64frombits(le.Uint64(b[:])), nil
	}
	return nil, fmt.Errorf("gguf: unexpected scalar type %d", t)
}

// skipArray reads past an array body without keeping its elements.
func (g ggufReader) skipArray(elem uint32, n uint64) error {
	size, err := fixedSize(elem)
	if err != nil {
		return err
	}
	switch elem {
	case ggufString:
		for i := uint64(0); i < n; i++ {
			l, err := g.u64()
			if err != nil {
				return err
			}
			if err := g.skip(l); err != nil {
				return err
			}
		}
		return nil
	case ggufArray:
		for i := uint64(0); i < n; i++ {
			et, err := g.u32()
			if err != nil {
				return err
			}
			en, err := g.u64()
			if err != nil {
				return err
			}
			if err := g.skipArray(et, en); err != nil {
				return err
			}
		}
		return nil
	}
	if n > math.MaxUint64/size {
		return fmt.Errorf("gguf: array too large")
	}
	return g.skip(n * size)
}

// intLike reports whether t is an integer or bool type.
func intLike(t uint32) bool {
	switch t {
	case ggufUint8, ggufInt8, ggufUint16, ggufInt16, ggufUint32, ggufInt32, ggufUint64, ggufInt64, ggufBool:
		return true
	}
	return false
}

// toInt converts a decoded integer or bool scalar to int64.
func toInt(v any) int64 {
	switch x := v.(type) {
	case uint64:
		return int64(x)
	case int64:
		return x
	case bool:
		if x {
			return 1
		}
	}
	return 0
}

// parseGGUF decodes the GGUF (v2 or v3, little endian) key/value header.
// If the input ends early it returns the keys decoded so far together with
// an error wrapping io.ErrUnexpectedEOF, so callers can retry with more bytes.
func parseGGUF(r io.Reader) (ggufMeta, error) {
	g := ggufReader{r: bufio.NewReaderSize(r, 1<<16)}
	meta := ggufMeta{KV: map[string]any{}}
	fail := func(err error) (ggufMeta, error) {
		if errors.Is(err, io.EOF) {
			err = io.ErrUnexpectedEOF
		}
		return meta, err
	}

	var magic [4]byte
	if _, err := io.ReadFull(g.r, magic[:]); err != nil {
		return fail(err)
	}
	if string(magic[:]) != "GGUF" {
		return meta, fmt.Errorf("gguf: bad magic %q", magic[:])
	}
	v, err := g.u32()
	if err != nil {
		return fail(err)
	}
	if v != 2 && v != 3 {
		return meta, fmt.Errorf("gguf: unsupported version %d", v)
	}
	meta.Version = v
	if _, err := g.u64(); err != nil { // tensor count
		return fail(err)
	}
	kvCount, err := g.u64()
	if err != nil {
		return fail(err)
	}
	for i := uint64(0); i < kvCount; i++ {
		key, err := g.str()
		if err != nil {
			return fail(err)
		}
		t, err := g.u32()
		if err != nil {
			return fail(err)
		}
		if t == ggufArray {
			et, err := g.u32()
			if err != nil {
				return fail(err)
			}
			n, err := g.u64()
			if err != nil {
				return fail(err)
			}
			if intLike(et) && n <= maxKeptArray {
				vals := make([]int64, n)
				for j := range vals {
					x, err := g.scalar(et)
					if err != nil {
						return fail(err)
					}
					vals[j] = toInt(x)
				}
				meta.KV[key] = vals
				continue
			}
			if err := g.skipArray(et, n); err != nil {
				return fail(err)
			}
			meta.KV[key] = ggufArrayInfo{ElemType: et, Len: n}
			continue
		}
		val, err := g.scalar(t)
		if err != nil {
			return fail(err)
		}
		meta.KV[key] = val
	}
	return meta, nil
}

// archInfo is the subset of GGUF metadata the catalog records.
type archInfo struct {
	Arch        string
	Layers      int
	KVHeads     int
	HeadDim     int
	MaxCtx      int
	ExpertCount int
	ExpertUsed  int
	FileType    uint64
	// Set only for hybrid or sliding-window attention (see kvSplit): K+V
	// elements per token in layers whose cache grows with the context, and in
	// sliding-window layers whose cache stops at SWAWindow tokens.
	KVFullElems int
	KVSWAElems  int
	SWAWindow   int
	// StateBytes is fixed memory for recurrent layers (see recurrentState).
	StateBytes uint64
}

// ollamaStateCopies is how many copies of the recurrent state Ollama keeps
// (it checkpoints the state for prompt caching). Measured on Ollama 0.24.0
// with qwen3.5:4b: a fixed 1.25 GiB at every context length, which is 25
// copies of the 50 MiB state the GGUF describes.
const ollamaStateCopies = 25

// archInfoFrom extracts the architecture fields, keyed by general.architecture.
// head_dim is attention.key_length, else embedding_length / head_count.
func archInfoFrom(m ggufMeta) (archInfo, error) {
	arch, ok := m.String("general.architecture")
	if !ok || arch == "" {
		return archInfo{}, errors.New("gguf: missing general.architecture")
	}
	get := func(k string) int {
		v, _ := m.Uint(arch + "." + k)
		return int(v)
	}
	info := archInfo{
		Arch:        arch,
		Layers:      get("block_count"),
		KVHeads:     get("attention.head_count_kv"),
		HeadDim:     get("attention.key_length"),
		MaxCtx:      get("context_length"),
		ExpertCount: get("expert_count"),
		ExpertUsed:  get("expert_used_count"),
	}
	info.FileType, _ = m.Uint("general.file_type")
	if info.HeadDim == 0 {
		if heads := get("attention.head_count"); heads > 0 {
			info.HeadDim = get("embedding_length") / heads
		}
	}
	if kvh, ok := m.Ints(arch + ".attention.head_count_kv"); ok { // per-layer
		for _, h := range kvh {
			info.KVHeads = max(info.KVHeads, int(h))
		}
	}
	if info.KVHeads == 0 { // head_count_kv absent means no GQA
		info.KVHeads = get("attention.head_count")
	}
	kvSplit(m, arch, &info)
	missing := ""
	switch {
	case info.Layers == 0:
		missing = "block_count"
	case info.KVHeads == 0:
		missing = "attention.head_count_kv"
	case info.HeadDim == 0:
		missing = "head_dim"
	case info.MaxCtx == 0:
		missing = "context_length"
	}
	if missing != "" {
		return info, fmt.Errorf("gguf: %s: missing %s", arch, missing)
	}
	return info, nil
}

// fileTypeNames maps llama.cpp LLAMA_FTYPE values to Ollama-style quant names.
var fileTypeNames = map[uint64]string{
	0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 7: "q8_0", 8: "q5_0", 9: "q5_1",
	10: "q2_K", 11: "q3_K_S", 12: "q3_K_M", 13: "q3_K_L", 14: "q4_K_S", 15: "q4_K_M",
	16: "q5_K_S", 17: "q5_K_M", 18: "q6_K", 19: "iq2_xxs", 20: "iq2_xs", 21: "q2_K_S",
	22: "iq3_xs", 23: "iq3_xxs", 24: "iq1_s", 25: "iq4_nl", 26: "iq3_s", 27: "iq3_m",
	28: "iq2_s", 29: "iq2_m", 30: "iq4_xs", 31: "iq1_m", 32: "bf16", 36: "tq1_0",
	37: "tq2_0", 38: "mxfp4",
}

// kvSplit fills the KV fields for models whose cache is not simply
// layers x kv_heads x head_dim: per-layer KV head counts, sliding-window
// layers (attention.sliding_window_pattern, true = sliding), recurrent layers
// with a fixed-size state instead of a KV cache (attention.recurrent_layers,
// or every layer but each full_attention_interval-th), and trailing layers
// that reuse an earlier layer's cache (attention.shared_kv_layers). Models
// with none of these keys keep the plain formula and get zeros here.
func kvSplit(m ggufMeta, arch string, info *archInfo) {
	k := func(s string) string { return arch + "." + s }
	get := func(s string) int { v, _ := m.Uint(k(s)); return int(v) }
	kvhArr, perLayer := m.Ints(k("attention.head_count_kv"))
	swaPat, hasSWA := m.Ints(k("attention.sliding_window_pattern"))
	rec, hasRec := m.Ints(k("attention.recurrent_layers"))
	interval := get("full_attention_interval")
	shared := get("attention.shared_kv_layers")
	window := get("attention.sliding_window")
	if !perLayer && !hasSWA && !hasRec && interval == 0 && shared == 0 {
		return
	}
	L := info.Layers
	kl, vl := info.HeadDim, get("attention.value_length")
	if vl == 0 {
		vl = kl
	}
	ks, vs := get("attention.key_length_swa"), get("attention.value_length_swa")
	if ks == 0 {
		ks = kl
	}
	if vs == 0 {
		vs = ks
	}
	at := func(a []int64, i int) int64 {
		if i < len(a) {
			return a[i]
		}
		return 0
	}
	recurrentLayers := 0
	for i := 0; i < L-shared; i++ {
		recurrent := at(rec, i) == 1 || (!hasRec && interval > 0 && (i+1)%interval != 0)
		if recurrent {
			recurrentLayers++
			continue
		}
		heads := info.KVHeads
		if perLayer && i < len(kvhArr) {
			heads = int(kvhArr[i])
		}
		if at(swaPat, i) == 1 && window > 0 {
			info.KVSWAElems += heads * (ks + vs)
		} else {
			info.KVFullElems += heads * (kl + vl)
		}
	}
	if info.KVSWAElems > 0 {
		info.SWAWindow = window
	}
	info.StateBytes = uint64(recurrentLayers) * recurrentState(get) * ollamaStateCopies
}

// recurrentState is the f32 state of one gated-delta-net / Mamba-style
// layer: the conv window, (conv_kernel-1) x (inner + 2 x groups x state),
// plus the recurrent matrix, state_size x inner.
func recurrentState(get func(string) int) uint64 {
	k, inner, groups, state := get("ssm.conv_kernel"), get("ssm.inner_size"), get("ssm.group_count"), get("ssm.state_size")
	if k == 0 || inner == 0 || state == 0 {
		return 0
	}
	return 4 * uint64((k-1)*(inner+2*groups*state)+state*inner)
}
