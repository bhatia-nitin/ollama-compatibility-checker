package estimate

import (
	"math"
	"testing"

	"github.com/shra26/ollama-compatibility-checker/internal/catalog"
	"github.com/shra26/ollama-compatibility-checker/internal/hw"
)

const gib = uint64(1) << 30

func model(size uint64, params, active float64) (catalog.Model, catalog.Variant) {
	v := catalog.Variant{Tag: "t:q4", SizeBytes: size, Bits: 4.5, Quant: "q4_K_M"}
	m := catalog.Model{
		Family: "f", Name: "m", ParamsB: params, ActiveParamsB: active,
		Layers: 32, KVHeads: 8, HeadDim: 128, MaxCtx: 131072, Quality: 70,
		KVQuantOK: true, Variants: []catalog.Variant{v},
	}
	return m, v
}

func cudaGPU(vram uint64, bw float64) hw.GPU {
	return hw.GPU{Vendor: "nvidia", Name: "gpu", VRAMBytes: vram, BandwidthGBs: bw, Backend: "cuda", Supported: true}
}

func near(t *testing.T, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol*want {
		t.Fatalf("got %v want %v (tol %v)", got, want, tol)
	}
}

func TestKVBytes(t *testing.T) {
	m := catalog.Model{Layers: 48, KVHeads: 8, HeadDim: 128}
	if got := KVBytes(m, 8192, true); got != 805306368 {
		t.Fatalf("q8 got %d", got)
	}
	if got := KVBytes(m, 8192, false); got != 2*805306368 {
		t.Fatalf("f16 got %d", got)
	}
}

func TestFullFitRTX4070(t *testing.T) {
	h := hw.Hardware{GPUs: []hw.GPU{cudaGPU(12*gib, 504)}, RAMBytes: 32 * gib, RAMBandwidthGBs: 50}
	m, v := model(8990000000, 8, 8)
	e := EstimateVariant(h, m, v, DefaultParams())
	if e.GPUFraction != 1 || e.NumGPU != 99 || !e.KVQuant || e.Excluded {
		t.Fatalf("unexpected %+v", e)
	}
	near(t, e.TokS, 0.65*504e9/8.99e9, 0.01)
}

func TestPartialOffload(t *testing.T) {
	h := hw.Hardware{GPUs: []hw.GPU{cudaGPU(8*gib, 300)}, RAMBytes: 32 * gib, RAMBandwidthGBs: 50}
	m, v := model(14000000000, 14, 14)
	p := DefaultParams()
	e := EstimateVariant(h, m, v, p)
	if e.GPUFraction <= 0 || e.GPUFraction >= 1 || e.Excluded {
		t.Fatalf("unexpected %+v", e)
	}
	kv := KVBytes(m, e.Ctx, e.KVQuant)
	if want := NumGPULayers(m, v, UsableVRAM(h), kv); e.NumGPU != want || e.NumGPU <= 0 || e.NumGPU >= m.Layers {
		t.Fatalf("NumGPU %d want %d", e.NumGPU, want)
	}
	cpu := p.EffCPU * 50e9 / 14e9
	gpu := p.EffGPU * 300e9 / 14e9
	if e.TokS <= cpu || e.TokS >= gpu {
		t.Fatalf("TokS %v not in (%v,%v)", e.TokS, cpu, gpu)
	}
}

func TestMoEFaster(t *testing.T) {
	h := hw.Hardware{GPUs: []hw.GPU{cudaGPU(12*gib, 504)}, RAMBytes: 32 * gib, RAMBandwidthGBs: 50}
	m, v := model(8000000000, 80, 80)
	dense := EstimateVariant(h, m, v, DefaultParams())
	m.ActiveParamsB = 10
	moe := EstimateVariant(h, m, v, DefaultParams())
	near(t, moe.TokS, 8*dense.TokS, 1e-9)
}

func TestAppleUnified(t *testing.T) {
	h := hw.Hardware{
		UnifiedMemory: true, RAMBytes: 32 * gib, RAMBandwidthGBs: 400,
		GPUs: []hw.GPU{{Vendor: "apple", VRAMBytes: 32 * gib * 66 / 100, BandwidthGBs: 400, Backend: "metal", Supported: true}},
	}
	// kv (q8, ctx 8192) = 512 MiB; overhead 512 MiB.
	m, v := model(23*gib, 30, 30)
	p := DefaultParams()
	e := EstimateVariant(h, m, v, p)
	if e.NeedBytes != 24*gib || e.Excluded || e.GPUFraction <= 0 || e.GPUFraction >= 1 {
		t.Fatalf("24GiB case %+v", e)
	}
	bpt := float64(23 * gib)
	want := 1 / (bpt*e.GPUFraction/(400e9*p.EffMetal) + bpt*(1-e.GPUFraction)/(400e9*p.EffCPU))
	near(t, e.TokS, want, 1e-9)

	m, v = model(26*gib, 30, 30)
	e = EstimateVariant(h, m, v, p)
	if e.NeedBytes != 27*gib || !e.Excluded {
		t.Fatalf("27GiB case %+v", e)
	}
}

func TestExcludedNonApple(t *testing.T) {
	h := hw.Hardware{GPUs: []hw.GPU{cudaGPU(8*gib, 300)}, RAMBytes: 16 * gib, RAMBandwidthGBs: 50}
	m, v := model(20*gib, 30, 30) // need 21 GiB > 7.2 + 12.8 = 20 GiB
	if e := EstimateVariant(h, m, v, DefaultParams()); !e.Excluded {
		t.Fatalf("want excluded %+v", e)
	}
	m, v = model(10*gib, 30, 30)
	if e := EstimateVariant(h, m, v, DefaultParams()); e.Excluded {
		t.Fatalf("want included %+v", e)
	}
}

func TestCPUOnly(t *testing.T) {
	h := hw.Hardware{RAMBytes: 32 * gib, RAMBandwidthGBs: 40}
	m, v := model(4000000000, 7, 7)
	e := EstimateVariant(h, m, v, DefaultParams())
	if e.GPUFraction != 0 || e.NumGPU != 0 || e.KVQuant || e.LowConfidence {
		t.Fatalf("unexpected %+v", e)
	}
	near(t, e.TokS, 0.55*40e9/4e9, 1e-9)
	if UsableVRAM(h) != 0 {
		t.Fatal("usable vram should be 0")
	}
}

func TestKVQuantNotOK(t *testing.T) {
	h := hw.Hardware{GPUs: []hw.GPU{cudaGPU(12*gib, 504)}, RAMBytes: 32 * gib, RAMBandwidthGBs: 50}
	m, v := model(4000000000, 7, 7)
	m.KVQuantOK = false
	e := EstimateVariant(h, m, v, DefaultParams())
	if e.KVQuant || e.NeedBytes != 4000000000+KVBytes(m, 8192, false)+512<<20 {
		t.Fatalf("unexpected %+v", e)
	}
}

func TestLowConfidencePropagates(t *testing.T) {
	g := cudaGPU(12*gib, 100)
	g.BandwidthLowConfidence = true
	h := hw.Hardware{GPUs: []hw.GPU{g}, RAMBytes: 32 * gib, RAMBandwidthGBs: 50}
	m, v := model(4000000000, 7, 7)
	if !EstimateVariant(h, m, v, DefaultParams()).LowConfidence {
		t.Fatal("want low confidence")
	}
}

func TestUsableVRAMFreeSmaller(t *testing.T) {
	g := cudaGPU(10*gib, 100)
	g.FreeVRAMBytes = 2 * gib
	if got := UsableVRAM(hw.Hardware{GPUs: []hw.GPU{g}}); got != 2*gib {
		t.Fatalf("got %d", got)
	}
}

func TestCtxClamp(t *testing.T) {
	h := hw.Hardware{GPUs: []hw.GPU{cudaGPU(12*gib, 504)}, RAMBytes: 32 * gib, RAMBandwidthGBs: 50}
	m, v := model(1000000000, 1, 1)
	m.MaxCtx = 4096
	e := EstimateVariant(h, m, v, DefaultParams())
	if e.Ctx != 4096 || !e.CtxReduced {
		t.Fatalf("unexpected %+v", e)
	}
}

func TestEstimateAllEmbedded(t *testing.T) {
	h := hw.Hardware{GPUs: []hw.GPU{cudaGPU(12*gib, 504)}, RAMBytes: 32 * gib, RAMBandwidthGBs: 50}
	ms := catalog.Embedded()
	all := EstimateAll(h, ms, DefaultParams())
	n := 0
	for _, m := range ms {
		n += len(m.Variants)
	}
	if len(all) != n {
		t.Fatalf("got %d estimates want %d", len(all), n)
	}
	full, excl := 0, 0
	for _, e := range all {
		if math.IsNaN(e.TokS) || math.IsInf(e.TokS, 0) || e.TokS <= 0 {
			t.Fatalf("bad TokS %v for %s", e.TokS, e.Variant.Tag)
		}
		if e.GPUFraction == 1 && !e.Excluded {
			full++
		}
		if e.Excluded {
			excl++
		}
	}
	if full == 0 || excl == 0 {
		t.Fatalf("full=%d excluded=%d", full, excl)
	}
}
