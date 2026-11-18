package main

import (
	"bytes"
	"testing"
)

func meta(kv map[string]any) ggufMeta { return ggufMeta{Version: 3, KV: kv} }

// Qwen3.5 4B as published: 32 layers, every 4th full attention, 4 KV heads
// of 256, gated delta net in the rest.
func TestKVSplitRecurrent(t *testing.T) {
	m := meta(map[string]any{
		"general.architecture": "qwen35", "qwen35.block_count": uint64(32), "qwen35.context_length": uint64(262144),
		"qwen35.attention.head_count_kv": uint64(4), "qwen35.attention.key_length": uint64(256),
		"qwen35.attention.value_length": uint64(256), "qwen35.full_attention_interval": uint64(4),
		"qwen35.ssm.conv_kernel": uint64(4), "qwen35.ssm.inner_size": uint64(4096),
		"qwen35.ssm.group_count": uint64(16), "qwen35.ssm.state_size": uint64(128),
	})
	info, err := archInfoFrom(m)
	if err != nil {
		t.Fatal(err)
	}
	if info.KVFullElems != 8*4*512 || info.KVSWAElems != 0 {
		t.Errorf("kv elems full %d swa %d", info.KVFullElems, info.KVSWAElems)
	}
	// 24 recurrent layers x 2,195,456 bytes x 25 copies (about 1.23 GiB,
	// against 1.25 GiB measured in Ollama 0.24.0).
	if want := uint64(24 * 2195456 * 25); info.StateBytes != want {
		t.Errorf("state %d, want %d", info.StateBytes, want)
	}
}

// Gemma 4 E2B-like: sliding window 5 of 6 layers, per-layer KV heads, last
// layers share KV.
func TestKVSplitSlidingShared(t *testing.T) {
	pattern := []int64{1, 1, 0, 1, 1, 0}
	m := meta(map[string]any{
		"general.architecture": "g", "g.block_count": uint64(6), "g.context_length": uint64(131072),
		"g.attention.head_count_kv": []int64{2, 2, 1, 2, 2, 1}, "g.attention.key_length": uint64(512),
		"g.attention.key_length_swa": uint64(256), "g.attention.sliding_window": uint64(512),
		"g.attention.sliding_window_pattern": pattern, "g.attention.shared_kv_layers": uint64(3),
	})
	info, err := archInfoFrom(m)
	if err != nil {
		t.Fatal(err)
	}
	// Layers 0-2 own a cache: two sliding (2 heads x 512) and one full (1 x 1024).
	if info.KVSWAElems != 2*2*512 || info.KVFullElems != 1*1024 || info.SWAWindow != 512 || info.KVHeads != 2 {
		t.Errorf("got %+v", info)
	}
	if info.StateBytes != 0 {
		t.Errorf("no recurrent layers, state %d", info.StateBytes)
	}
}

func TestKVSplitPlainModelUnchanged(t *testing.T) {
	m := meta(map[string]any{
		"general.architecture": "llama", "llama.block_count": uint64(32), "llama.context_length": uint64(8192),
		"llama.attention.head_count_kv": uint64(8), "llama.attention.key_length": uint64(128),
		"llama.attention.sliding_window": uint64(4096), // no pattern: not treated as hybrid
	})
	info, err := archInfoFrom(m)
	if err != nil {
		t.Fatal(err)
	}
	if info.KVFullElems != 0 || info.KVSWAElems != 0 || info.StateBytes != 0 {
		t.Errorf("plain model got hybrid fields: %+v", info)
	}
}

func TestParseKeepsSmallIntArrays(t *testing.T) {
	w := &ggufWriter{}
	w.buf.WriteString("GGUF")
	w.u32(3)
	w.u64(0)
	w.u64(1)
	w.str("x.pattern")
	w.u32(ggufArray)
	w.u32(ggufBool)
	w.u64(3)
	w.buf.Write([]byte{1, 0, 1})
	m, err := parseGGUF(bytes.NewReader(w.buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	got, ok := m.Ints("x.pattern")
	if !ok || len(got) != 3 || got[0] != 1 || got[1] != 0 || got[2] != 1 {
		t.Errorf("got %v %v", got, ok)
	}
}

// Nemotron-H / LFM2 style: per-layer head_count_kv with zeros marks layers
// that have no attention (Mamba or conv) and keep a fixed state instead.
func TestKVSplitZeroHeadsAreRecurrent(t *testing.T) {
	m := meta(map[string]any{
		"general.architecture": "nh", "nh.block_count": uint64(4), "nh.context_length": uint64(8192),
		"nh.attention.head_count_kv": []int64{0, 8, 0, 0}, "nh.attention.key_length": uint64(128),
		"nh.ssm.conv_kernel": uint64(4), "nh.ssm.inner_size": uint64(1024),
		"nh.ssm.group_count": uint64(8), "nh.ssm.state_size": uint64(128),
	})
	info, err := archInfoFrom(m)
	if err != nil {
		t.Fatal(err)
	}
	if info.KVFullElems != 8*256 {
		t.Errorf("full elems %d, want %d", info.KVFullElems, 8*256)
	}
	if want := 3 * recurrentState(func(k string) int {
		v, _ := m.Uint("nh." + k)
		return int(v)
	}); info.StateBytes != want { // llama.cpp-runner architecture: one copy
		t.Errorf("state %d, want %d", info.StateBytes, want)
	}
}

func TestKVSplitQwen3NextDefaultInterval(t *testing.T) {
	m := meta(map[string]any{
		"general.architecture": "qwen3next", "qwen3next.block_count": uint64(48), "qwen3next.context_length": uint64(262144),
		"qwen3next.attention.head_count_kv": uint64(2), "qwen3next.attention.key_length": uint64(256),
		"qwen3next.ssm.conv_kernel": uint64(4), "qwen3next.ssm.inner_size": uint64(4096),
		"qwen3next.ssm.group_count": uint64(16), "qwen3next.ssm.state_size": uint64(128),
	})
	info, _ := archInfoFrom(m)
	if info.KVFullElems != 12*2*512 || info.StateBytes == 0 {
		t.Errorf("got full %d state %d", info.KVFullElems, info.StateBytes)
	}
}

func TestKVSplitNextNLayerSkipped(t *testing.T) {
	m := meta(map[string]any{
		"general.architecture": "q", "q.block_count": uint64(65), "q.context_length": uint64(262144),
		"q.attention.head_count_kv": uint64(4), "q.attention.key_length": uint64(256),
		"q.full_attention_interval": uint64(4), "q.nextn_predict_layers": uint64(1),
	})
	info, _ := archInfoFrom(m)
	if info.KVFullElems != 16*4*512 {
		t.Errorf("full %d, want %d", info.KVFullElems, 16*4*512)
	}
}

func TestKVSplitMLA(t *testing.T) {
	m := meta(map[string]any{
		"general.architecture": "g", "g.block_count": uint64(47), "g.context_length": uint64(202752),
		"g.attention.head_count_kv": uint64(20), "g.attention.key_length": uint64(256),
		"g.attention.kv_lora_rank": uint64(512), "g.attention.key_length_mla": uint64(576),
	})
	info, _ := archInfoFrom(m)
	if info.KVFullElems != 47*576 || info.KVSWAElems != 0 {
		t.Errorf("got %+v", info)
	}
}

func TestKVSplitChunked(t *testing.T) {
	m := meta(map[string]any{
		"general.architecture": "llama4", "llama4.block_count": uint64(48), "llama4.context_length": uint64(10485760),
		"llama4.attention.head_count_kv": uint64(8), "llama4.attention.key_length": uint64(128),
		"llama4.attention.chunk_size": uint64(8192),
	})
	info, _ := archInfoFrom(m)
	if info.KVFullElems != 12*8*256 || info.KVSWAElems != 36*8*256 || info.SWAWindow != 8192 {
		t.Errorf("got %+v", info)
	}
}
