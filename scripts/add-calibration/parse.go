package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"unicode"

	"github.com/shra26/ollama-compatibility-checker/internal/calib"
)

const (
	maxBody  = 64 << 10 // issue bodies larger than this are rejected
	maxRuns  = 50
	maxStr   = 120   // longest allowed string field
	maxTokS  = 10000 // decode tok/s above this is not a real measurement
	maxBWGBs = 20000
)

// fenced matches the first ```json (or bare ```) block, as the issue form's
// `render: json` textarea produces.
var fenced = regexp.MustCompile("(?s)```(?:json)?\\s*\\n(.*?)\\n\\s*```")

// parseIssue extracts and checks the record in an issue body.
func parseIssue(body []byte) (calib.Record, error) {
	var r calib.Record
	if len(body) > maxBody {
		return r, fmt.Errorf("issue body is larger than %d KB", maxBody>>10)
	}
	js := bytes.TrimSpace(body)
	if m := fenced.FindSubmatch(body); m != nil {
		js = m[1]
	} else if i := bytes.IndexByte(js, '{'); i >= 0 {
		js = js[i:]
		if j := bytes.LastIndexByte(js, '}'); j >= 0 {
			js = js[:j+1]
		}
	}
	dec := json.NewDecoder(bytes.NewReader(js))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&r); err != nil {
		return r, fmt.Errorf("could not read the calibration record JSON: %w", err)
	}
	if err := r.Validate(); err != nil {
		return r, fmt.Errorf("invalid record: %w", err)
	}
	return r, sane(r)
}

// sane rejects values no real machine produces and text that does not
// belong in a hardware name.
func sane(r calib.Record) error {
	strs := []string{r.ToolVersion, r.OllamaVersion, r.OS, r.Arch, r.CPU}
	if r.GPU != nil {
		strs = append(strs, r.GPU.Vendor, r.GPU.Name, r.GPU.Backend)
		if r.GPU.BandwidthGBs > maxBWGBs {
			return fmt.Errorf("gpu bandwidth %v GB/s is not plausible", r.GPU.BandwidthGBs)
		}
	}
	if r.RAMBandwidthGBs > maxBWGBs {
		return fmt.Errorf("ram bandwidth %v GB/s is not plausible", r.RAMBandwidthGBs)
	}
	if len(r.Runs) > maxRuns {
		return fmt.Errorf("%d runs; at most %d", len(r.Runs), maxRuns)
	}
	for _, u := range r.Runs {
		strs = append(strs, u.Tag)
		if u.MeasuredTokS > maxTokS || u.EstTokS > maxTokS {
			return fmt.Errorf("run %s: tok/s above %d is not plausible", u.Tag, maxTokS)
		}
		if u.EstGPUFraction < 0 || u.EstGPUFraction > 1 || u.MeasuredGPUFraction < 0 || u.MeasuredGPUFraction > 1 {
			return fmt.Errorf("run %s: GPU fractions must be between 0 and 1", u.Tag)
		}
	}
	for _, s := range strs {
		if len(s) > maxStr {
			return fmt.Errorf("field %.20q... is longer than %d characters", s, maxStr)
		}
		for _, c := range s {
			if !unicode.IsPrint(c) || c == '<' || c == '>' || c == '`' {
				return fmt.Errorf("field %q has a character that is not allowed", s)
			}
		}
	}
	return nil
}

// encode writes the record the way bench --share does.
func encode(r calib.Record) ([]byte, error) {
	js, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(js, '\n'), nil
}

var nonSlug = regexp.MustCompile(`[^a-z0-9]+`)

// fileName is YYYY-MM-DD-<hardware>-<ram>gb-issue<N>.json, hardware being the
// GPU name, else the CPU.
func fileName(r calib.Record, issue int) string {
	hw := r.CPU
	if r.GPU != nil && r.GPU.Name != "" {
		hw = r.GPU.Name
	}
	slug := strings.Trim(nonSlug.ReplaceAllString(strings.ToLower(hw), "-"), "-")
	if slug == "" {
		slug = "unknown"
	}
	if len(slug) > 40 {
		slug = strings.TrimRight(slug[:40], "-")
	}
	return fmt.Sprintf("%s-%s-%dgb-issue%d.json", r.Date, slug, (r.RAMBytes+1<<29)>>30, issue)
}
