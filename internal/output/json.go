// Package output renders hardware reports, recommendations and the model
// catalog as text or JSON.
package output

import (
	"encoding/json"
	"io"

	"github.com/shra26/ollama-compatibility-checker/internal/estimate"
	"github.com/shra26/ollama-compatibility-checker/internal/hw"
	"github.com/shra26/ollama-compatibility-checker/internal/recommend"
)

// SchemaVersion is the version of the JSON documents below.
const SchemaVersion = 1

// WriteJSON writes v as indented JSON followed by a newline.
func WriteJSON(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}

type gpuJSON struct {
	Vendor                 string  `json:"vendor"`
	Name                   string  `json:"name"`
	VRAMBytes              uint64  `json:"vram_bytes"`
	FreeVRAMBytes          uint64  `json:"free_vram_bytes"`
	BandwidthGBs           float64 `json:"bandwidth_gbs"`
	BandwidthLowConfidence bool    `json:"bandwidth_low_confidence"`
	Backend                string  `json:"backend"`
	Supported              bool    `json:"supported"`
	Driver                 string  `json:"driver"`
	ComputeCap             string  `json:"compute_cap"`
}

type cpuJSON struct {
	Model   string `json:"model"`
	Cores   int    `json:"cores"`
	Threads int    `json:"threads"`
	AVX2    bool   `json:"avx2"`
	AVX512  bool   `json:"avx512"`
}

type hardwareJSON struct {
	OS              string    `json:"os"`
	Arch            string    `json:"arch"`
	GPUs            []gpuJSON `json:"gpus"`
	RAMBytes        uint64    `json:"ram_bytes"`
	RAMBandwidthGBs float64   `json:"ram_bandwidth_gbs"`
	CPU             cpuJSON   `json:"cpu"`
	UnifiedMemory   bool      `json:"unified_memory"`
	Warnings        []string  `json:"warnings"`
}

type estimateJSON struct {
	Family        string  `json:"family"`
	Name          string  `json:"name"`
	Tag           string  `json:"tag"`
	Quant         string  `json:"quant"`
	SizeBytes     uint64  `json:"size_bytes"`
	Ctx           int     `json:"ctx"`
	CtxReduced    bool    `json:"ctx_reduced"`
	KVQuant       bool    `json:"kv_quant"`
	NeedBytes     uint64  `json:"need_bytes"`
	GPUFraction   float64 `json:"gpu_fraction"`
	NumGPU        int     `json:"num_gpu"`
	TokS          float64 `json:"tok_s"`
	LowConfidence bool    `json:"low_confidence"`
}

type pickJSON struct {
	Kind      string         `json:"kind"`
	Met       bool           `json:"met"`
	Best      *estimateJSON  `json:"best"`
	RunnersUp []estimateJSON `json:"runners_up"`
	Closest   *estimateJSON  `json:"closest"`
	Note      string         `json:"note"`
}

type configJSON struct {
	PullCmd   string            `json:"pull_cmd"`
	Modelfile string            `json:"modelfile"`
	NumCtx    int               `json:"num_ctx"`
	NumGPU    int               `json:"num_gpu"`
	CtxNote   string            `json:"ctx_note"`
	Env       map[string]string `json:"env"`
	EnvSyntax map[string]string `json:"env_syntax"`
}

// ScanDocument is the JSON shape of `scan --json`.
type ScanDocument struct {
	SchemaVersion int          `json:"schema_version"`
	Hardware      hardwareJSON `json:"hardware"`
}

// RecommendDocument is the JSON shape of `recommend --json`.
type RecommendDocument struct {
	SchemaVersion int          `json:"schema_version"`
	Hardware      hardwareJSON `json:"hardware"`
	Picks         []pickJSON   `json:"picks"`
	Config        *configJSON  `json:"config"`
}

// ScanDoc maps hardware to its JSON document.
func ScanDoc(h hw.Hardware) ScanDocument {
	return ScanDocument{SchemaVersion: SchemaVersion, Hardware: mapHardware(h)}
}

// RecommendDoc maps a recommendation run to its JSON document. Config is
// null when cfg is empty (no candidate fit at all).
func RecommendDoc(h hw.Hardware, picks []recommend.Pick, cfg recommend.Config) RecommendDocument {
	d := RecommendDocument{SchemaVersion: SchemaVersion, Hardware: mapHardware(h), Picks: []pickJSON{}}
	for _, p := range picks {
		pj := pickJSON{Kind: p.Kind, Met: p.Met, Note: p.Note, RunnersUp: []estimateJSON{}}
		if p.Best != nil {
			e := mapEstimate(*p.Best)
			pj.Best = &e
		}
		if p.Closest != nil {
			e := mapEstimate(*p.Closest)
			pj.Closest = &e
		}
		for _, r := range p.RunnersUp {
			pj.RunnersUp = append(pj.RunnersUp, mapEstimate(r))
		}
		d.Picks = append(d.Picks, pj)
	}
	if cfg.PullCmd != "" {
		env := cfg.Env
		if env == nil {
			env = map[string]string{}
		}
		d.Config = &configJSON{PullCmd: cfg.PullCmd, Modelfile: cfg.Modelfile, NumCtx: cfg.NumCtx,
			NumGPU: cfg.NumGPU, CtxNote: cfg.CtxNote, Env: env, EnvSyntax: cfg.EnvSyntax}
	}
	return d
}

func mapHardware(h hw.Hardware) hardwareJSON {
	out := hardwareJSON{
		OS: h.OS, Arch: h.Arch, RAMBytes: h.RAMBytes, RAMBandwidthGBs: h.RAMBandwidthGBs,
		CPU:           cpuJSON{Model: h.CPU.Model, Cores: h.CPU.Cores, Threads: h.CPU.Threads, AVX2: h.CPU.AVX2, AVX512: h.CPU.AVX512},
		UnifiedMemory: h.UnifiedMemory, GPUs: []gpuJSON{}, Warnings: append([]string{}, h.Warnings...),
	}
	for _, g := range h.GPUs {
		out.GPUs = append(out.GPUs, gpuJSON{Vendor: g.Vendor, Name: g.Name, VRAMBytes: g.VRAMBytes,
			FreeVRAMBytes: g.FreeVRAMBytes, BandwidthGBs: g.BandwidthGBs, BandwidthLowConfidence: g.BandwidthLowConfidence,
			Backend: g.Backend, Supported: g.Supported, Driver: g.Driver, ComputeCap: g.ComputeCap})
	}
	return out
}

func mapEstimate(e estimate.Estimate) estimateJSON {
	return estimateJSON{Family: e.Model.Family, Name: e.Model.Name, Tag: e.Variant.Tag, Quant: e.Variant.Quant,
		SizeBytes: e.Variant.SizeBytes, Ctx: e.Ctx, CtxReduced: e.CtxReduced, KVQuant: e.KVQuant,
		NeedBytes: e.NeedBytes, GPUFraction: e.GPUFraction, NumGPU: e.NumGPU, TokS: e.TokS,
		LowConfidence: e.LowConfidence}
}
