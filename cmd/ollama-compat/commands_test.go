package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shra26/ollama-compatibility-checker/internal/hw"
)

var update = flag.Bool("update", false, "rewrite golden files")

const gb = 1 << 30

func fakeDetect(context.Context) hw.Hardware {
	return hw.Hardware{
		OS: "linux", Arch: "amd64", RAMBytes: 32 * gb, RAMBandwidthGBs: 60,
		CPU: hw.CPU{Model: "AMD Ryzen 7 7700X", Cores: 8, Threads: 16, AVX2: true, AVX512: true},
		GPUs: []hw.GPU{{Vendor: "nvidia", Name: "NVIDIA GeForce RTX 4070", VRAMBytes: 12 * gb, BandwidthGBs: 504,
			BandwidthLowConfidence: true, Backend: "cuda", Supported: true, Driver: "560.94"}},
	}
}

func run(t *testing.T, args ...string) (string, string, error) {
	t.Helper()
	var out, errb bytes.Buffer
	cmd := newRootCmdWith(&out, &errb, fakeDetect)
	cmd.SetArgs(args)
	err := cmd.Execute()
	return out.String(), errb.String(), err
}

func golden(t *testing.T, name, got string) {
	t.Helper()
	path := filepath.Join("testdata", name+".golden")
	if *update {
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden (run with -update): %v", err)
	}
	if string(want) != got {
		t.Errorf("%s mismatch\n--- want\n%s\n--- got\n%s", name, want, got)
	}
}

const cat = "testdata/catalog.json"

func TestGoldenCommands(t *testing.T) {
	cases := map[string][]string{
		"root":      {"--catalog", cat},
		"recommend": {"recommend", "--catalog", cat, "--top", "1", "--ctx", "4096"},
		"scan":      {"scan"},
		"catalog":   {"catalog", "--catalog", cat},
	}
	for name, args := range cases {
		out, _, err := run(t, args...)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		golden(t, name, out)
	}
}

func TestRootEqualsRecommend(t *testing.T) {
	a, _, _ := run(t, "--catalog", cat)
	b, _, _ := run(t, "recommend", "--catalog", cat)
	if a != b {
		t.Errorf("root and recommend differ:\n%s\n---\n%s", a, b)
	}
}

func TestJSONOutputs(t *testing.T) {
	for _, args := range [][]string{{"--json"}, {"recommend", "--json"}, {"scan", "--json"}} {
		out, _, err := run(t, args...)
		if err != nil {
			t.Fatal(err)
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(out), &m); err != nil {
			t.Fatalf("%v: invalid json: %v\n%s", args, err, out)
		}
		if m["schema_version"] != float64(1) {
			t.Errorf("%v: schema_version = %v", args, m["schema_version"])
		}
	}
	out, _, err := run(t, "catalog", "--json", "--catalog", cat)
	if err != nil {
		t.Fatal(err)
	}
	var ms []map[string]any
	if err := json.Unmarshal([]byte(out), &ms); err != nil || len(ms) != 4 {
		t.Errorf("catalog --json: err=%v len=%d", err, len(ms))
	}
}

func TestCatalogFlagErrors(t *testing.T) {
	if _, _, err := run(t, "--catalog", "testdata/missing.json"); err == nil {
		t.Error("missing catalog file: want error")
	}
	bad := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(bad, []byte(`[{"family":"x","name":"x","variants":[]}]`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := run(t, "--catalog", bad); err == nil {
		t.Error("invalid catalog: want error")
	}
}

func TestFlagValidation(t *testing.T) {
	for _, args := range [][]string{{"--ctx", "0"}, {"--top", "-1"}, {"--min-tps", "-5"}} {
		if _, _, err := run(t, args...); err == nil {
			t.Errorf("%v: want error", args)
		}
	}
}

type failTransport struct{ t *testing.T }

func (f failTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	f.t.Errorf("network request made: %s", r.URL)
	return nil, http.ErrUseLastResponse
}

func TestNoNetworkInDefaultRun(t *testing.T) {
	old := http.DefaultTransport
	http.DefaultTransport = failTransport{t}
	defer func() { http.DefaultTransport = old }()
	out, _, err := run(t)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "tok/s ±") {
		t.Errorf("default run produced no recommendations:\n%s", out)
	}
}
