package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/shra26/ollama-compatibility-checker/internal/bench"
	"github.com/shra26/ollama-compatibility-checker/internal/calib"
	"github.com/shra26/ollama-compatibility-checker/internal/catalog"
	"github.com/shra26/ollama-compatibility-checker/internal/estimate"
	"github.com/shra26/ollama-compatibility-checker/internal/hw"
)

func TestCalibrationRecord(t *testing.T) {
	h := hw.Hardware{OS: "linux", Arch: "amd64", RAMBytes: 32 << 30, RAMBandwidthGBs: 80, CPU: hw.CPU{Model: "cpu"},
		GPUs: []hw.GPU{{Name: "iGPU"}, {Vendor: "nvidia", Name: "RTX", Backend: "cuda", BandwidthGBs: 500, Supported: true}}}
	moe := estimate.Estimate{Model: catalog.Model{ParamsB: 30, ActiveParamsB: 3}, Variant: catalog.Variant{SizeBytes: 18e9}, GPUFraction: 1, TokS: 50}
	results := []bench.Result{
		{Tag: "moe:q4", Est: moe, DecodeTokS: 40, MeasuredGPUFraction: 1},
		{Tag: "failed:q4", Est: moe, Err: "boom"},
	}
	r := calibrationRecord(h, results, "0.24.0", time.Date(2026, 11, 17, 23, 0, 0, 0, time.UTC))
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	if r.GPU == nil || r.GPU.Name != "RTX" || len(r.Runs) != 1 || r.Runs[0].ActiveRatio != 0.1 || r.Date != "2026-11-17" {
		t.Fatalf("got %+v", r)
	}
}

func TestWriteShare(t *testing.T) {
	var msg bytes.Buffer
	if err := writeShare(&msg, filepath.Join(t.TempDir(), "x.json"), calib.Record{}); err == nil {
		t.Fatal("want error for a record with no runs")
	}
	path := filepath.Join(t.TempDir(), "rec.json")
	r := calib.Record{Schema: calib.Schema, Runs: []calib.Run{{Tag: "a"}}}
	if err := writeShare(&msg, path, r); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var back calib.Record
	if err := json.Unmarshal(raw, &back); err != nil || back.Runs[0].Tag != "a" {
		t.Fatalf("round trip: %v %+v", err, back)
	}
	if !strings.Contains(msg.String(), "Nothing was sent") || !strings.Contains(msg.String(), shareIssueURL) {
		t.Errorf("message:\n%s", msg.String())
	}
}
