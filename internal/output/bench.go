package output

import (
	"fmt"
	"io"
	"sort"
	"text/tabwriter"

	"github.com/shra26/ollama-compatibility-checker/internal/bench"
	"github.com/shra26/ollama-compatibility-checker/internal/hw"
)

const gpuMismatchNote = "GPU offload differs from estimate"

type benchResultJSON struct {
	Tag                 string  `json:"tag"`
	EstTokS             float64 `json:"est_tok_s"`
	MeasuredTokS        float64 `json:"measured_tok_s"`
	PrefillTokS         float64 `json:"prefill_tok_s"`
	EstGPUFraction      float64 `json:"est_gpu_fraction"`
	MeasuredGPUFraction float64 `json:"measured_gpu_fraction"`
	Mismatch            bool    `json:"mismatch"`
	Pulled              bool    `json:"pulled"`
	Error               string  `json:"error"`
}

// BenchDocument is the JSON shape of `bench --json`.
type BenchDocument struct {
	SchemaVersion int               `json:"schema_version"`
	Hardware      hardwareJSON      `json:"hardware"`
	Results       []benchResultJSON `json:"results"`
}

// BenchDoc maps benchmark results to their JSON document.
func BenchDoc(h hw.Hardware, results []bench.Result) BenchDocument {
	d := BenchDocument{SchemaVersion: SchemaVersion, Hardware: mapHardware(h), Results: []benchResultJSON{}}
	for _, r := range results {
		d.Results = append(d.Results, benchResultJSON{
			Tag: r.Tag, EstTokS: r.Est.TokS, MeasuredTokS: r.DecodeTokS, PrefillTokS: r.PrefillTokS,
			EstGPUFraction: r.Est.GPUFraction, MeasuredGPUFraction: r.MeasuredGPUFraction,
			Mismatch: r.Mismatch, Pulled: r.Pulled, Error: r.Err,
		})
	}
	return d
}

// RenderBench prints the estimate versus measured table and a ranking by
// measured decode speed.
func RenderBench(w io.Writer, results []bench.Result) {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "Model\test. tok/s\tmeasured tok/s\tprefill tok/s\tGPU est/measured\tNote")
	var ok []bench.Result
	for _, r := range results {
		note := r.Err
		if r.Err == "" {
			ok = append(ok, r)
			if r.Mismatch {
				note = gpuMismatchNote
			}
			fmt.Fprintf(tw, "%s\t%.1f\t%.1f\t%.1f\t%.0f%%/%.0f%%\t%s\n", r.Tag, r.Est.TokS, r.DecodeTokS,
				r.PrefillTokS, r.Est.GPUFraction*100, r.MeasuredGPUFraction*100, note)
			continue
		}
		fmt.Fprintf(tw, "%s\t%.1f\t-\t-\t-\t%s\n", r.Tag, r.Est.TokS, note)
	}
	_ = tw.Flush()
	if len(ok) == 0 {
		return
	}
	sort.SliceStable(ok, func(i, j int) bool { return ok[i].DecodeTokS > ok[j].DecodeTokS })
	fmt.Fprintln(w, "\nMeasured ranking (decode tok/s):")
	for i, r := range ok {
		fmt.Fprintf(w, "  %d. %s  %.1f tok/s measured (est. %.1f)\n", i+1, r.Tag, r.DecodeTokS, r.Est.TokS)
	}
}
