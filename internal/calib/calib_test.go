package calib

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func rec(name, backend string, bw float64, lowConf bool, runs ...Run) Record {
	return Record{Schema: Schema, Date: "2026-11-17", RAMBytes: 8 << 30, RAMBandwidthGBs: 68,
		GPU: &GPU{Name: name, Backend: backend, BandwidthGBs: bw, BandwidthLowConfidence: lowConf}, Runs: runs}
}

// run returns a fully-on-GPU run with efficiency eff at 100 GB/s and 1 GB size.
func run(eff float64) Run {
	return Run{Tag: "m:q4", SizeBytes: 1e9, ActiveRatio: 1, MeasuredTokS: eff * 100, EstGPUFraction: 1, MeasuredGPUFraction: 1}
}

func TestFit(t *testing.T) {
	partial := run(0.9)
	partial.EstGPUFraction = 0.5
	rs := []Record{
		rec("A", "cuda", 100, false, run(0.60), run(0.70), partial),
		rec("B", "cuda", 100, false, run(0.50)),
		rec("C", "cuda", 100, true, run(0.10)), // guessed bandwidth: ignored
		rec("M", "metal", 100, false, run(0.57)),
	}
	got := Fit(rs)
	want := Table{
		// median 0.60; worst miss 0.5/0.6 = 16.7% -> 20
		"cuda":  {Eff: 0.6, ErrPct: 20, Runs: 3, Machines: 2},
		"metal": {Eff: 0.57, ErrPct: MinErrPct, Runs: 1, Machines: 1},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got  %+v\nwant %+v", got, want)
	}
}

func TestValidate(t *testing.T) {
	ok := rec("A", "cuda", 100, false, run(0.6))
	if err := ok.Validate(); err != nil {
		t.Fatal(err)
	}
	for name, mut := range map[string]func(*Record){
		"schema":   func(r *Record) { r.Schema = 2 },
		"date":     func(r *Record) { r.Date = "17/11/2026" },
		"no runs":  func(r *Record) { r.Runs = nil },
		"ratio":    func(r *Record) { r.Runs[0].ActiveRatio = 1.5 },
		"measured": func(r *Record) { r.Runs[0].MeasuredTokS = 0 },
		"gpu bw":   func(r *Record) { r.GPU.BandwidthGBs = 0 },
	} {
		r := rec("A", "cuda", 100, false, run(0.6))
		mut(&r)
		if r.Validate() == nil {
			t.Errorf("%s: want error", name)
		}
	}
}

// The embedded factors must equal a fresh fit of calibration/data. If this
// fails, run `go run ./scripts/fit-calibration` and commit factors.json.
func TestEmbeddedFactorsUpToDate(t *testing.T) {
	paths, err := filepath.Glob("../../calibration/data/*.json")
	if err != nil {
		t.Fatal(err)
	}
	var rs []Record
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		var r Record
		if err := json.Unmarshal(raw, &r); err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		if err := r.Validate(); err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		r.RefreshBandwidth()
		rs = append(rs, r)
	}
	if got, want := Factors(), Fit(rs); !reflect.DeepEqual(got, want) {
		t.Fatalf("factors.json is stale; run `go run ./scripts/fit-calibration`\nembedded %+v\nfitted   %+v", got, want)
	}
}

func TestRefreshBandwidth(t *testing.T) {
	// Shared by tool 0.1.3, which guessed laptop bandwidth as half the desktop part.
	r := Record{GPU: &GPU{Vendor: "nvidia", Name: "NVIDIA GeForce RTX 3070 Ti Laptop GPU", Backend: "cuda",
		VRAMBytes: 8 << 30, BandwidthGBs: 304, BandwidthLowConfidence: true}}
	r.RefreshBandwidth()
	if r.GPU.BandwidthGBs != 448 || r.GPU.BandwidthLowConfidence {
		t.Errorf("got %v GB/s low=%v", r.GPU.BandwidthGBs, r.GPU.BandwidthLowConfidence)
	}
	m := Record{GPU: &GPU{Name: "Apple M1", Backend: "metal", BandwidthGBs: 1}}
	m.RefreshBandwidth()
	if m.GPU.BandwidthGBs != 68.25 {
		t.Errorf("apple: got %v", m.GPU.BandwidthGBs)
	}
}
