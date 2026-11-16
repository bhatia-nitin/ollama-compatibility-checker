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

// ggufMeta is the decoded key/value header. Unsigned integers are stored as
// uint64, signed as int64, floats as float64; arrays as ggufArrayInfo.
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
}

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
	if info.KVHeads == 0 { // head_count_kv absent means no GQA
		info.KVHeads = get("attention.head_count")
	}
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
