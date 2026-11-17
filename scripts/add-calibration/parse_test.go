package main

import (
	"os"
	"strings"
	"testing"
)

func seed(t *testing.T) string {
	t.Helper()
	raw, err := os.ReadFile("../../calibration/data/2026-11-17-apple-m1-8gb.json")
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

// formBody is what GitHub's issue form produces for calibration.yml.
func formBody(record string) string {
	return "### Calibration record\n\n```json\n" + record + "\n```\n\n### Anything unusual? (optional)\n\n_No response_\n"
}

func TestParseIssueFormBody(t *testing.T) {
	r, err := parseIssue([]byte(formBody(seed(t))))
	if err != nil {
		t.Fatal(err)
	}
	if r.GPU == nil || r.GPU.Name != "Apple M1" || len(r.Runs) != 1 {
		t.Fatalf("got %+v", r)
	}
	if got := fileName(r, 7); got != "2026-11-17-apple-m1-8gb-issue7.json" {
		t.Errorf("file name %q", got)
	}
	// Re-encoding reproduces the bench --share file byte for byte.
	js, err := encode(r)
	if err != nil || string(js) != seed(t) {
		t.Errorf("re-encoded record differs from the shared file")
	}
}

func TestParseIssueBareJSON(t *testing.T) {
	if _, err := parseIssue([]byte("here you go:\n" + seed(t) + "\nthanks")); err != nil {
		t.Fatal(err)
	}
}

func TestParseIssueRejects(t *testing.T) {
	s := seed(t)
	for name, body := range map[string]string{
		"no json":       "I ran it, it was fast",
		"unknown field": strings.Replace(s, `"schema": 1,`, `"schema": 1, "hostname": "my-laptop",`, 1),
		"bad schema":    strings.Replace(s, `"schema": 1`, `"schema": 9`, 1),
		"huge tok/s":    strings.Replace(s, `"measured_tok_s": 20.09220295611647`, `"measured_tok_s": 99999`, 1),
		"markup name":   strings.Replace(s, `"name": "Apple M1"`, `"name": "<img src=x onerror=alert(1)>"`, 1),
		"backtick":      strings.Replace(s, `"cpu": "Apple M1"`, "\"cpu\": \"$(`whoami`)\"", 1),
		"fraction":      strings.Replace(s, `"measured_gpu_fraction": 1`, `"measured_gpu_fraction": 3`, 1),
		"too big":       formBody(s) + strings.Repeat("x", maxBody),
	} {
		if _, err := parseIssue([]byte(formBody(body))); err == nil {
			t.Errorf("%s: want error", name)
		}
	}
}

func TestFileNameSlug(t *testing.T) {
	r, err := parseIssue([]byte(formBody(seed(t))))
	if err != nil {
		t.Fatal(err)
	}
	r.GPU.Name = "NVIDIA GeForce RTX 4070 Ti SUPER / ../../etc"
	r.RAMBytes = 32 << 30
	if got := fileName(r, 12); got != "2026-11-17-nvidia-geforce-rtx-4070-ti-super-etc-32gb-issue12.json" {
		t.Errorf("got %q", got)
	}
}
