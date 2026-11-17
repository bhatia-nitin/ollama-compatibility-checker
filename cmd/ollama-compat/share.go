package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/shra26/ollama-compatibility-checker/internal/bench"
	"github.com/shra26/ollama-compatibility-checker/internal/calib"
	"github.com/shra26/ollama-compatibility-checker/internal/hw"
)

const shareIssueURL = "https://github.com/shra26/ollama-compatibility-checker/issues/new?template=calibration.yml"

// calibrationRecord turns successful bench results into a shareable record.
// It carries model names and speeds only, no host-identifying data.
func calibrationRecord(h hw.Hardware, results []bench.Result, ollamaVersion string, now time.Time) calib.Record {
	r := calib.Record{
		Schema: calib.Schema, Date: now.UTC().Format("2006-01-02"),
		ToolVersion: version, OllamaVersion: ollamaVersion,
		OS: h.OS, Arch: h.Arch, CPU: h.CPU.Model,
		RAMBytes: h.RAMBytes, RAMBandwidthGBs: h.RAMBandwidthGBs, UnifiedMemory: h.UnifiedMemory,
		Runs: []calib.Run{},
	}
	for _, g := range h.GPUs {
		if g.Supported {
			r.GPU = &calib.GPU{Vendor: g.Vendor, Name: g.Name, Backend: g.Backend, VRAMBytes: g.VRAMBytes,
				BandwidthGBs: g.BandwidthGBs, BandwidthLowConfidence: g.BandwidthLowConfidence}
			break
		}
	}
	for _, res := range results {
		if res.Err != "" || res.DecodeTokS <= 0 {
			continue
		}
		ratio := 1.0
		if m := res.Est.Model; m.ParamsB > 0 && m.ActiveParamsB > 0 {
			ratio = m.ActiveParamsB / m.ParamsB
		}
		r.Runs = append(r.Runs, calib.Run{
			Tag: res.Tag, SizeBytes: res.Est.Variant.SizeBytes, ActiveRatio: ratio, Ctx: res.Est.Ctx,
			EstTokS: res.Est.TokS, MeasuredTokS: res.DecodeTokS, PrefillTokS: res.PrefillTokS,
			EstGPUFraction: res.Est.GPUFraction, MeasuredGPUFraction: res.MeasuredGPUFraction,
		})
	}
	return r
}

// writeShare saves the record to path and tells the user how to contribute it.
func writeShare(msg io.Writer, path string, r calib.Record) error {
	if len(r.Runs) == 0 {
		return fmt.Errorf("--share: no successful runs to record")
	}
	js, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, append(js, '\n'), 0o644); err != nil {
		return fmt.Errorf("--share: %w", err)
	}
	fmt.Fprintf(msg, "\nSaved calibration record to %s (hardware model names and speeds only;\n", path)
	fmt.Fprintln(msg, "no hostname, username or paths). Nothing was sent. To contribute it, open")
	fmt.Fprintln(msg, "  "+shareIssueURL)
	fmt.Fprintln(msg, "and paste the file. Each record tightens the speed estimates for everyone on that hardware.")
	return nil
}
