// Package recommend chooses the Fastest, Balanced and Largest picks and
// builds copy-paste Ollama configuration for them.
package recommend

import (
	"fmt"
	"sort"

	"github.com/shra26/ollama-compatibility-checker/internal/catalog"
	"github.com/shra26/ollama-compatibility-checker/internal/estimate"
	"github.com/shra26/ollama-compatibility-checker/internal/hw"
)

const (
	fastestMinScore = 40 // keeps toy models out of the Fastest pick
	largestMinTPS   = 5  // usable but slow
	// fullGPUGain is the speedup a full-GPU fit must bring to be worth a
	// near-miss line. On unified memory it is usually small: CPU and GPU
	// share one memory bus.
	fullGPUGain = 1.25
)

// Options controls the picks.
type Options struct {
	MinTPS   float64
	Ctx, Top int
	Runtime  string // "" or RuntimeOllama, RuntimeLlamaCpp, RuntimeLMStudio
	Task     string // "" for all models, else one of catalog.Tasks
}

// Pick is one recommendation slot. When Met is false, Best is nil and
// Closest holds the highest-tok/s candidate (nil if nothing fits at all).
type Pick struct {
	Kind      string // "fastest" | "balanced" | "largest"
	Best      *estimate.Estimate
	RunnersUp []estimate.Estimate
	Met       bool
	Closest   *estimate.Estimate
	Note      string
}

// Score is the model quality adjusted for quantization.
func Score(e estimate.Estimate) int {
	return e.Model.Quality + catalog.QuantPenalty(e.Variant.Quant)
}

// Recommend returns exactly three picks: fastest, balanced, largest.
func Recommend(h hw.Hardware, ms []catalog.Model, o Options) []Pick {
	p := estimate.DefaultParams()
	p.Ctx = o.Ctx
	var cands []estimate.Estimate
	for _, e := range estimate.EstimateAll(h, catalog.FilterTask(ms, o.Task), p) {
		if !e.Excluded {
			cands = append(cands, e)
		}
	}
	return []Pick{pickFastest(cands, o), pickBalanced(cands, o), pickLargest(cands, o)}
}

func filter(cands []estimate.Estimate, keep func(estimate.Estimate) bool) []estimate.Estimate {
	var out []estimate.Estimate
	for _, e := range cands {
		if keep(e) {
			out = append(out, e)
		}
	}
	return out
}

func byTokS(a, b estimate.Estimate) bool { return a.TokS > b.TokS }

func byScore(a, b estimate.Estimate) bool {
	if sa, sb := Score(a), Score(b); sa != sb {
		return sa > sb
	}
	return a.TokS > b.TokS
}

// pickFrom ranks set by better, takes the top as Best and the next o.Top
// (excluding Best's tag) as runners-up.
func pickFrom(kind string, set []estimate.Estimate, better func(a, b estimate.Estimate) bool, top int, note string) Pick {
	s := append([]estimate.Estimate(nil), set...)
	sort.SliceStable(s, func(i, j int) bool { return better(s[i], s[j]) })
	best := s[0]
	pk := Pick{Kind: kind, Best: &best, Met: true, Note: note}
	for _, e := range s[1:] {
		if len(pk.RunnersUp) >= top {
			break
		}
		if e.Variant.Tag != best.Variant.Tag {
			pk.RunnersUp = append(pk.RunnersUp, e)
		}
	}
	return pk
}

func unmet(kind string, cands []estimate.Estimate, note string) Pick {
	if len(cands) == 0 {
		return Pick{Kind: kind, Note: "No model in the catalog fits this machine's memory."}
	}
	s := append([]estimate.Estimate(nil), cands...)
	sort.SliceStable(s, func(i, j int) bool { return byTokS(s[i], s[j]) })
	c := s[0]
	return Pick{Kind: kind, Closest: &c, Note: note}
}

func pickFastest(cands []estimate.Estimate, o Options) Pick {
	if set := filter(cands, func(e estimate.Estimate) bool { return e.GPUFraction == 1 && Score(e) >= fastestMinScore }); len(set) > 0 {
		return pickFrom("fastest", set, byTokS, o.Top, "")
	}
	if set := filter(cands, func(e estimate.Estimate) bool { return Score(e) >= fastestMinScore }); len(set) > 0 {
		return pickFrom("fastest", set, byTokS, o.Top,
			"Because nothing fits fully on GPU here, this is the fastest model that runs partly or wholly on CPU.")
	}
	return unmet("fastest", cands, fmt.Sprintf("No model with a quality score of %d or more fits this machine; closest shown.", fastestMinScore))
}

func pickBalanced(cands []estimate.Estimate, o Options) Pick {
	set := filter(cands, func(e estimate.Estimate) bool { return e.TokS >= o.MinTPS })
	if len(set) == 0 {
		return unmet("balanced", cands, fmt.Sprintf("No model reaches %g tok/s on this hardware; closest shown.", o.MinTPS))
	}
	return pickFrom("balanced", set, byScore, o.Top, "")
}

func pickLargest(cands []estimate.Estimate, o Options) Pick {
	set := filter(cands, func(e estimate.Estimate) bool { return e.TokS >= largestMinTPS })
	if len(set) == 0 {
		return unmet("largest", cands, fmt.Sprintf("No model reaches %d tok/s on this hardware; closest shown.", largestMinTPS))
	}
	return pickFrom("largest", set, byScore, o.Top, "")
}

// Upgrade is the Largest pick this machine would get with twice the memory
// that limits it: unified memory on Apple Silicon, VRAM on a discrete GPU,
// RAM when there is no GPU.
type Upgrade struct {
	MemBytes uint64 // the doubled memory size
	MemKind  string // "unified memory" | "VRAM" | "RAM"
	Largest  estimate.Estimate
	// FullGPU is set when Largest is the current Largest model, which would
	// move from partial offload to running fully on GPU.
	FullGPU bool
}

// NearMiss returns the Upgrade when doubling memory unlocks a higher-scoring
// Largest pick, or lets the current one run fully on GPU; else nil.
func NearMiss(h hw.Hardware, ms []catalog.Model, o Options, current []Pick) *Upgrade {
	up, kind, size := doubled(h)
	var cur *estimate.Estimate
	for _, p := range current {
		if p.Kind == "largest" && p.Met {
			cur = p.Best
		}
	}
	for _, p := range Recommend(up, ms, o) {
		if p.Kind != "largest" || !p.Met {
			continue
		}
		switch {
		case cur == nil || Score(*p.Best) > Score(*cur):
			return &Upgrade{MemBytes: size, MemKind: kind, Largest: *p.Best}
		case p.Best.Variant.Tag == cur.Variant.Tag && cur.GPUFraction < 1 && p.Best.GPUFraction == 1 &&
			p.Best.TokS >= fullGPUGain*cur.TokS:
			return &Upgrade{MemBytes: size, MemKind: kind, Largest: *p.Best, FullGPU: true}
		}
	}
	return nil
}

// doubled copies h with its limiting memory pool doubled. Bandwidth is kept.
func doubled(h hw.Hardware) (hw.Hardware, string, uint64) {
	up := h
	up.GPUs = append([]hw.GPU(nil), h.GPUs...)
	gi := -1
	for i, g := range up.GPUs {
		if g.Supported {
			gi = i
			break
		}
	}
	switch {
	case h.UnifiedMemory:
		up.RAMBytes *= 2
		if gi >= 0 {
			up.GPUs[gi].VRAMBytes *= 2
			up.GPUs[gi].FreeVRAMBytes = 0
		}
		return up, "unified memory", up.RAMBytes
	case gi >= 0:
		up.GPUs[gi].VRAMBytes *= 2
		up.GPUs[gi].FreeVRAMBytes = 0
		return up, "VRAM", up.GPUs[gi].VRAMBytes
	default:
		up.RAMBytes *= 2
		return up, "RAM", up.RAMBytes
	}
}
