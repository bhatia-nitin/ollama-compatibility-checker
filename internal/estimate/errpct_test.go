package estimate

import (
	"testing"

	"github.com/shra26/ollama-compatibility-checker/internal/calib"
	"github.com/shra26/ollama-compatibility-checker/internal/catalog"
	"github.com/shra26/ollama-compatibility-checker/internal/hw"
)

func TestErrPct(t *testing.T) {
	p := Params{EffGPU: 0.65, EffCPU: 0.55, Fitted: calib.Table{"metal": {Eff: 0.6, ErrPct: 15}}}
	for _, c := range []struct {
		backend string
		frac    float64
		low     bool
		want    int
	}{
		{"metal", 1, false, 15},   // fitted
		{"cuda", 1, false, 30},    // not fitted
		{"", 0, false, 30},        // CPU only
		{"metal", 0.5, false, 40}, // partial offload
		{"metal", 1, true, 50},    // guessed bandwidth
	} {
		if got := p.errPct(c.backend, c.frac, c.low); got != c.want {
			t.Errorf("errPct(%q, %v, %v) = %d, want %d", c.backend, c.frac, c.low, got, c.want)
		}
	}
	lo, hi := Estimate{TokS: 20, ErrPct: 15}.TokSRange()
	if lo != 17 || hi != 23 {
		t.Errorf("range = %v..%v, want 17..23", lo, hi)
	}
}

func TestGPUEffUsesFittedFactor(t *testing.T) {
	p := Params{EffGPU: 0.65, Fitted: calib.Table{"metal": {Eff: 0.58, ErrPct: 15}}}
	if got := p.gpuEff(hwGPU("metal")); got != 0.58 {
		t.Errorf("metal = %v, want fitted 0.58", got)
	}
	if got := p.gpuEff(hwGPU("cuda")); got != 0.65 {
		t.Errorf("cuda = %v, want default 0.65", got)
	}
}

func hwGPU(backend string) hw.GPU { return hw.GPU{Backend: backend, Supported: true} }

func TestKVBytesHybrid(t *testing.T) {
	m := catalog.Model{Layers: 32, KVHeads: 4, HeadDim: 256, KVFullElems: 16384, KVSWAElems: 1000, SWAWindow: 512}
	// full grows with ctx, sliding stops at the window; q8 = 1 byte per element
	if got, want := KVBytes(m, 8192, true), uint64(16384*8192+1000*512); got != want {
		t.Errorf("KVBytes = %d, want %d", got, want)
	}
	if got, want := KVBytes(m, 256, false), uint64((16384*256+1000*256)*2); got != want {
		t.Errorf("short ctx = %d, want %d", got, want)
	}
}

func TestNeedBytesCountsProjectorAndState(t *testing.T) {
	m := catalog.Model{Name: "m", ParamsB: 4, ActiveParamsB: 4, Layers: 32, KVHeads: 4, HeadDim: 256, MaxCtx: 32768,
		KVFullElems: 16384, StateBytes: 1 << 30}
	v := catalog.Variant{Tag: "m:q4", SizeBytes: 2 << 30, ExtraBytes: 700 << 20, Quant: "q4_K_M"}
	h := hw.Hardware{RAMBytes: 64 << 30, RAMBandwidthGBs: 100}
	e := EstimateVariant(h, m, v, Params{Ctx: 8192, EffGPU: 0.65, EffCPU: 0.55})
	if want := v.SizeBytes + v.ExtraBytes + m.StateBytes + KVBytes(m, 8192, false) + runtimeOverhead; e.NeedBytes != want {
		t.Errorf("NeedBytes = %d, want %d", e.NeedBytes, want)
	}
}
