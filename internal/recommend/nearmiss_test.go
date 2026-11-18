package recommend

import (
	"testing"

	"github.com/shra26/ollama-compatibility-checker/internal/catalog"
)

var nmOpts = Options{MinTPS: 20, Ctx: 8192, Top: 3}

func TestNearMissUnlocksBiggerModel(t *testing.T) {
	ms := append(synth(), cm("giant", 20_000_000_000, 95))
	h := cudaHW()
	picks := Recommend(h, ms, nmOpts)
	if picks[2].Best.Model.Name == "giant" {
		t.Fatal("fixture: giant should not be Largest at 12 GiB")
	}
	up := NearMiss(h, ms, nmOpts, picks)
	if up == nil || up.Largest.Model.Name != "giant" || up.MemKind != "VRAM" || up.MemBytes != 24*gib {
		t.Fatalf("got %+v", up)
	}
	if h.GPUs[0].VRAMBytes != 12*gib {
		t.Fatal("NearMiss mutated the caller's hardware")
	}
}

func TestNearMissNilWhenNothingNew(t *testing.T) {
	h := cudaHW()
	ms := synth()[:4] // all fit fully on GPU already
	if up := NearMiss(h, ms, nmOpts, Recommend(h, ms, nmOpts)); up != nil {
		t.Fatalf("want nil, got %+v", up)
	}
}

func TestNearMissUnifiedDoublesRAM(t *testing.T) {
	h := cudaHW()
	h.UnifiedMemory, h.RAMBytes = true, 8*gib
	h.GPUs[0].Backend, h.GPUs[0].VRAMBytes = "metal", 5*gib
	up, kind, size := doubled(h)
	if kind != "unified memory" || size != 16*gib || up.GPUs[0].VRAMBytes != 10*gib || h.RAMBytes != 8*gib {
		t.Fatalf("got %s %d %+v", kind, size, up)
	}
}

func TestNearMissFullGPUForSameModel(t *testing.T) {
	ms := []catalog.Model{cm("small", 2_000_000_000, 45), cm("huge", 16_000_000_000, 90)}
	h := cudaHW()
	picks := Recommend(h, ms, nmOpts)
	if picks[2].Best.Model.Name != "huge" || picks[2].Best.GPUFraction == 1 {
		t.Fatalf("fixture: want huge partially offloaded, got %+v", picks[2].Best)
	}
	up := NearMiss(h, ms, nmOpts, picks)
	if up == nil || !up.FullGPU || up.Largest.Model.Name != "huge" || up.Largest.GPUFraction != 1 {
		t.Fatalf("got %+v", up)
	}
}
