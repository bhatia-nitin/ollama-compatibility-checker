// Command fit-calibration reads every calibration record in a directory,
// validates it and writes the fitted per-backend factors that the estimator
// embeds. Run from the repo root:
//
//	go run ./scripts/fit-calibration
//
// CI runs it and fails when the committed factors.json is out of date.
package main

import (
	"encoding/json"
	"flag"
	"log"
	"os"
	"path/filepath"
	"sort"

	"github.com/shra26/ollama-compatibility-checker/internal/calib"
)

func main() {
	in := flag.String("in", "calibration/data", "directory of calibration records (*.json)")
	out := flag.String("out", "internal/calib/factors.json", "fitted factors output")
	flag.Parse()
	log.SetFlags(0)

	paths, err := filepath.Glob(filepath.Join(*in, "*.json"))
	if err != nil {
		log.Fatal(err)
	}
	sort.Strings(paths)
	var rs []calib.Record
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			log.Fatal(err)
		}
		var r calib.Record
		if err := json.Unmarshal(raw, &r); err != nil {
			log.Fatalf("%s: %v", p, err)
		}
		if err := r.Validate(); err != nil {
			log.Fatalf("%s: %v", p, err)
		}
		r.RefreshBandwidth()
		if r.GPU != nil && r.GPU.BandwidthLowConfidence {
			log.Printf("%s: %s bandwidth is a guess; its runs are not fitted (add it to internal/hw/bandwidth.go)", p, r.GPU.Name)
		}
		rs = append(rs, r)
	}
	t := calib.Fit(rs)
	js, err := json.MarshalIndent(t, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(*out, append(js, '\n'), 0o644); err != nil {
		log.Fatal(err)
	}
	for b, f := range t {
		log.Printf("%s: eff %.3f, ±%d%%, %d runs on %d machines", b, f.Eff, f.ErrPct, f.Runs, f.Machines)
	}
	log.Printf("wrote %s from %d records", *out, len(rs))
}
