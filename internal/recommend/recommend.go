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
)

// Options controls the picks.
type Options struct {
	MinTPS   float64
	Ctx, Top int
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
	for _, e := range estimate.EstimateAll(h, ms, p) {
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
