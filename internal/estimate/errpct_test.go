package estimate

import (
	"testing"

	"github.com/shra26/ollama-compatibility-checker/internal/calib"
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
