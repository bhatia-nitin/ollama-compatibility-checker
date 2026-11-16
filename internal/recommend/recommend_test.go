package recommend

import (
	"strings"
	"testing"

	"github.com/shra26/ollama-compatibility-checker/internal/catalog"
	"github.com/shra26/ollama-compatibility-checker/internal/hw"
)

const gib = uint64(1) << 30

func cm(name string, size uint64, quality int) catalog.Model {
	return catalog.Model{
		Family: "f", Name: name, ParamsB: 8, ActiveParamsB: 8,
		Layers: 32, KVHeads: 8, HeadDim: 128, MaxCtx: 131072, Quality: quality,
		KVQuantOK: true,
		Variants:  []catalog.Variant{{Tag: name + ":q4", SizeBytes: size, Bits: 4.5, Quant: "q4_K_M"}},
	}
}

func cudaHW() hw.Hardware {
	return hw.Hardware{
		GPUs:     []hw.GPU{{Vendor: "nvidia", Name: "g", VRAMBytes: 12 * gib, BandwidthGBs: 500, Backend: "cuda", Supported: true}},
		RAMBytes: 32 * gib, RAMBandwidthGBs: 100,
	}
}

// Scores (quality-3): tiny 22, small 42, mid 67, big 77, huge 87.
func synth() []catalog.Model {
	return []catalog.Model{
		cm("tiny", 400_000_000, 25),
		cm("small", 2_000_000_000, 45),
		cm("mid", 5_000_000_000, 70),
		cm("big", 9_000_000_000, 80),
		cm("huge", 16_000_000_000, 90), // partial offload, about 6 tok/s
	}
}

func TestFastestSkipsToyModels(t *testing.T) {
	p := Recommend(cudaHW(), synth(), Options{MinTPS: 20, Ctx: 8192, Top: 3})
	if len(p) != 3 || p[0].Kind != "fastest" || p[1].Kind != "balanced" || p[2].Kind != "largest" {
		t.Fatalf("bad picks %+v", p)
	}
	f := p[0]
	if !f.Met || f.Best == nil || f.Best.Model.Name != "small" {
		t.Fatalf("fastest = %+v", f.Best)
	}
	for _, r := range f.RunnersUp {
		if r.Model.Name == "tiny" {
			t.Fatal("toy model in runners-up")
		}
	}
}

func TestBalancedPicksHighestQualityAboveMinTPS(t *testing.T) {
	p := Recommend(cudaHW(), synth(), Options{MinTPS: 20, Ctx: 8192, Top: 3})[1]
	if !p.Met || p.Best.Model.Name != "big" {
		t.Fatalf("balanced = %+v", p.Best)
	}
	p = Recommend(cudaHW(), synth(), Options{MinTPS: 50, Ctx: 8192, Top: 3})[1]
	if !p.Met || p.Best.Model.Name != "mid" {
		t.Fatalf("balanced@50 = %+v", p.Best)
	}
}

func TestLargestAllowsPartialAboveFive(t *testing.T) {
	p := Recommend(cudaHW(), synth(), Options{MinTPS: 20, Ctx: 8192, Top: 3})[2]
	if !p.Met || p.Best.Model.Name != "huge" {
		t.Fatalf("largest = %+v", p.Best)
	}
	if p.Best.GPUFraction >= 1 || p.Best.TokS < 5 {
		t.Fatalf("expected partial offload >= 5 tok/s, got %+v", p.Best)
	}
}

func TestNothingMeetsThreshold(t *testing.T) {
	h := hw.Hardware{RAMBytes: 4 * gib, RAMBandwidthGBs: 5}
	ms := []catalog.Model{cm("a", 1_000_000_000, 30), cm("b", 1_500_000_000, 35)}
	for _, p := range Recommend(h, ms, Options{MinTPS: 20, Ctx: 8192, Top: 3}) {
		if p.Met || p.Best != nil || p.Closest == nil || p.Note == "" {
			t.Fatalf("%s: %+v", p.Kind, p)
		}
	}
	// Nothing fits memory at all.
	for _, p := range Recommend(h, []catalog.Model{cm("x", 40_000_000_000, 90)}, Options{MinTPS: 20, Ctx: 8192}) {
		if p.Met || p.Best != nil || p.Closest != nil || !strings.Contains(p.Note, "fits this machine's memory") {
			t.Fatalf("%s: %+v", p.Kind, p)
		}
	}
}

func TestMessagePhrasing(t *testing.T) {
	h := hw.Hardware{RAMBytes: 4 * gib, RAMBandwidthGBs: 5}
	p := Recommend(h, []catalog.Model{cm("a", 1_000_000_000, 45)}, Options{MinTPS: 20, Ctx: 8192})
	if p[1].Note != "No model reaches 20 tok/s on this hardware; closest shown." {
		t.Fatalf("note %q", p[1].Note)
	}
}

func TestFastestFallsBackToPartial(t *testing.T) {
	h := cudaHW()
	h.GPUs[0].VRAMBytes = 1 * gib
	p := Recommend(h, []catalog.Model{cm("small", 2_000_000_000, 45)}, Options{MinTPS: 1, Ctx: 8192})[0]
	if !p.Met || p.Best == nil || !strings.Contains(p.Note, "nothing fits fully on GPU") {
		t.Fatalf("%+v", p)
	}
}

func TestRunnersUpCount(t *testing.T) {
	p := Recommend(cudaHW(), synth(), Options{MinTPS: 20, Ctx: 8192, Top: 3})[1]
	// Balanced candidates >= 20 tok/s: big(77), mid(67), small(42), tiny(22): best + 3 runners.
	if len(p.RunnersUp) != 3 {
		t.Fatalf("runners %d", len(p.RunnersUp))
	}
	for i := 1; i < len(p.RunnersUp); i++ {
		if Score(p.RunnersUp[i]) > Score(p.RunnersUp[i-1]) {
			t.Fatal("runners-up not sorted by score")
		}
	}
	for _, r := range p.RunnersUp {
		if r.Variant.Tag == p.Best.Variant.Tag {
			t.Fatal("best repeated")
		}
	}
	p = Recommend(cudaHW(), synth(), Options{MinTPS: 20, Ctx: 8192, Top: 1})[1]
	if len(p.RunnersUp) != 1 {
		t.Fatalf("runners %d", len(p.RunnersUp))
	}
	p = Recommend(cudaHW(), synth(), Options{MinTPS: 20, Ctx: 8192, Top: 0})[1]
	if len(p.RunnersUp) != 0 {
		t.Fatalf("runners %d", len(p.RunnersUp))
	}
}

func TestEmbeddedCPUOnly4GiB(t *testing.T) {
	h := hw.Hardware{RAMBytes: 4 * gib, RAMBandwidthGBs: 5}
	ps := Recommend(h, catalog.Embedded(), Options{MinTPS: 20, Ctx: 8192, Top: 3})
	if len(ps) != 3 {
		t.Fatalf("got %d picks", len(ps))
	}
	for _, p := range ps {
		if p.Met && p.Best == nil {
			t.Fatalf("%s: met with nil best", p.Kind)
		}
		if !p.Met && (p.Best != nil || p.Note == "") {
			t.Fatalf("%s: bad unmet pick %+v", p.Kind, p)
		}
	}
}
