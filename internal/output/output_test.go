package output

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/shra26/ollama-compatibility-checker/internal/catalog"
	"github.com/shra26/ollama-compatibility-checker/internal/hw"
	"github.com/shra26/ollama-compatibility-checker/internal/recommend"
)

var update = flag.Bool("update", false, "rewrite golden files")

const gb = 1 << 30

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

func testCatalog(t *testing.T) []catalog.Model {
	t.Helper()
	f, err := os.Open("testdata/catalog.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	ms, err := catalog.Load(f)
	if err != nil {
		t.Fatal(err)
	}
	if err := catalog.Validate(ms); err != nil {
		t.Fatal(err)
	}
	return ms
}

func gpuHW() hw.Hardware {
	return hw.Hardware{
		OS: "linux", Arch: "amd64", RAMBytes: 32 * gb, RAMBandwidthGBs: 60,
		CPU: hw.CPU{Model: "AMD Ryzen 7 7700X", Cores: 8, Threads: 16, AVX2: true, AVX512: true},
		GPUs: []hw.GPU{
			{Vendor: "nvidia", Name: "NVIDIA GeForce RTX 4070", VRAMBytes: 12 * gb, BandwidthGBs: 504,
				BandwidthLowConfidence: true, Backend: "cuda", Supported: true, Driver: "560.94"},
			{Vendor: "intel", Name: "Intel UHD Graphics", VRAMBytes: 1 * gb, BandwidthGBs: 30, Backend: "vulkan"},
		},
		Warnings: []string{"nvidia-smi reported no free VRAM figure"},
	}
}

func cpuHW() hw.Hardware {
	return hw.Hardware{
		OS: "linux", Arch: "amd64", RAMBytes: 16 * gb, RAMBandwidthGBs: 40,
		CPU: hw.CPU{Model: "Intel Core i5-12400", Cores: 6, Threads: 12, AVX2: true},
	}
}

func slowHW() hw.Hardware {
	h := cpuHW()
	h.RAMBytes, h.RAMBandwidthGBs = 6*gb, 8
	return h
}

var opts = recommend.Options{MinTPS: 20, Ctx: 8192, Top: 3}

func renderRec(t *testing.T, h hw.Hardware, o recommend.Options) string {
	t.Helper()
	picks := recommend.Recommend(h, testCatalog(t), o)
	var cfg recommend.Config
	if e := ConfigTarget(picks); e != nil {
		cfg = recommend.BuildConfig(h, *e, o)
	}
	var b bytes.Buffer
	RenderRecommend(&b, h, picks, cfg, o, recommend.NearMiss(h, testCatalog(t), o, picks), false)
	return b.String()
}

func TestGoldens(t *testing.T) {
	var scan bytes.Buffer
	RenderScan(&scan, gpuHW(), false)
	golden(t, "scan", scan.String())
	golden(t, "recommend", renderRec(t, gpuHW(), opts))
	golden(t, "recommend_cpu", renderRec(t, cpuHW(), opts))
	golden(t, "recommend_unmet", renderRec(t, slowHW(), opts))
	var cat bytes.Buffer
	RenderCatalog(&cat, testCatalog(t))
	golden(t, "catalog", cat.String())
}

func TestRecommendAssertions(t *testing.T) {
	for name, h := range map[string]hw.Hardware{"gpu": gpuHW(), "cpu": cpuHW(), "slow": slowHW()} {
		out := renderRec(t, h, opts)
		for _, line := range strings.Split(out, "\n") {
			if strings.Contains(line, "~") && strings.Contains(line, "tok/s") && !regexp.MustCompile(`tok/s ±\d+%`).MatchString(line) {
				t.Errorf("%s: tok/s without error range: %q", name, line)
			}
		}
		if strings.ContainsRune(out, 0x2014) {
			t.Errorf("%s: em dash in output", name)
		}
	}
	gpu := renderRec(t, gpuHW(), opts)
	for _, want := range []string{"low confidence", "not used by Ollama", "% GPU", "! nvidia-smi"} {
		if !strings.Contains(gpu, want) {
			t.Errorf("gpu output missing %q", want)
		}
	}
	cpu := renderRec(t, cpuHW(), opts)
	for _, want := range []string{"No supported GPU; estimates use CPU and system RAM.", "CPU"} {
		if !strings.Contains(cpu, want) {
			t.Errorf("cpu output missing %q", want)
		}
	}
	if strings.Contains(cpu, "% GPU") {
		t.Error("cpu-only output should not show % GPU")
	}
	slow := renderRec(t, slowHW(), opts)
	if !strings.Contains(slow, "No model reaches 20 tok/s") || !strings.Contains(slow, "closest") {
		t.Errorf("unmet output missing note/closest:\n%s", slow)
	}
	if !strings.Contains(slow, "Config for Balanced (closest)") {
		t.Errorf("unmet config heading wrong:\n%s", slow)
	}
}

func TestFooterSize(t *testing.T) {
	h := gpuHW()
	picks := recommend.Recommend(h, testCatalog(t), opts)
	seen := map[string]bool{}
	var total uint64
	for _, p := range picks {
		if p.Met && !seen[p.Best.Variant.Tag] {
			seen[p.Best.Variant.Tag] = true
			total += p.Best.Variant.SizeBytes
		}
	}
	want := fmt.Sprintf("Run `ollama-compat bench` to measure these (downloads up to ~%.1f GB, asks first).", float64(total)/1e9)
	if out := renderRec(t, h, opts); !strings.Contains(out, want) {
		t.Errorf("footer %q not in output:\n%s", want, out)
	}
}

func TestEnvSyntaxPerOS(t *testing.T) {
	cases := []struct {
		goos      string
		want, not []string
	}{
		{"darwin", []string{"export OLLAMA_FLASH_ATTENTION=1", "for the Ollama.app", "launchctl setenv"}, []string{"setx", "$env:"}},
		{"linux", []string{"export OLLAMA_FLASH_ATTENTION=1"}, []string{"launchctl", "setx", "$env:"}},
		{"windows", []string{"$env:OLLAMA_FLASH_ATTENTION", "persistent", "setx OLLAMA_FLASH_ATTENTION 1"}, []string{"export ", "launchctl"}},
	}
	for _, c := range cases {
		h := gpuHW()
		h.OS = c.goos
		if c.goos == "darwin" {
			h.GPUs = []hw.GPU{{Vendor: "apple", Name: "Apple M2", VRAMBytes: 24 * gb, BandwidthGBs: 100, Backend: "metal", Supported: true}}
			h.UnifiedMemory, h.RAMBytes = true, 32*gb
		}
		out := renderRec(t, h, opts)
		for _, w := range c.want {
			if !strings.Contains(out, w) {
				t.Errorf("%s: missing %q\n%s", c.goos, w, out)
			}
		}
		for _, n := range c.not {
			if strings.Contains(out, n) {
				t.Errorf("%s: unexpected %q", c.goos, n)
			}
		}
	}
	if out := renderRec(t, cpuHW(), opts); strings.Contains(out, "env") {
		t.Errorf("cpu-only has no env, got:\n%s", out)
	}
}

func TestColor(t *testing.T) {
	var b bytes.Buffer
	RenderScan(&b, gpuHW(), true)
	if !strings.Contains(b.String(), "\x1b[") {
		t.Error("color=true should emit ANSI")
	}
	b.Reset()
	RenderScan(&b, gpuHW(), false)
	if strings.Contains(b.String(), "\x1b[") {
		t.Error("color=false must not emit ANSI")
	}
}

func TestJSONDocs(t *testing.T) {
	h := gpuHW()
	picks := recommend.Recommend(h, testCatalog(t), opts)
	e := ConfigTarget(picks)
	cfg := recommend.BuildConfig(h, *e, opts)
	var b bytes.Buffer
	if err := WriteJSON(&b, RecommendDoc(h, picks, cfg)); err != nil {
		t.Fatal(err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b.Bytes(), &m); err != nil {
		t.Fatal(err)
	}
	if string(m["schema_version"]) != "1" {
		t.Errorf("schema_version = %s", m["schema_version"])
	}
	for _, k := range []string{"hardware", "picks", "config"} {
		if _, ok := m[k]; !ok {
			t.Errorf("missing key %q", k)
		}
	}
	if !regexp.MustCompile(`"est_tok_s"`).Match(b.Bytes()) || !regexp.MustCompile(`"pull_cmd"`).Match(b.Bytes()) {
		t.Error("expected snake_case keys est_tok_s and pull_cmd")
	}
	b.Reset()
	if err := WriteJSON(&b, ScanDoc(h)); err != nil {
		t.Fatal(err)
	}
	m = nil
	if err := json.Unmarshal(b.Bytes(), &m); err != nil {
		t.Fatal(err)
	}
	if string(m["schema_version"]) != "1" || len(m) != 2 {
		t.Errorf("scan doc keys = %v", m)
	}
}

func TestMetalNoteOnlyForMetal(t *testing.T) {
	const note = "Note: Apple Silicon estimates are calibrated on one M1 8 GB; other chips may differ. Run `ollama-compat bench` to measure."
	mac := hw.Hardware{
		OS: "darwin", Arch: "arm64", RAMBytes: 16 * gb, RAMBandwidthGBs: 100, UnifiedMemory: true,
		CPU:  hw.CPU{Model: "Apple M1", Cores: 8, Threads: 8},
		GPUs: []hw.GPU{{Vendor: "apple", Name: "Apple M1", VRAMBytes: 10 * gb, BandwidthGBs: 68, Backend: "metal", Supported: true}},
	}
	got := renderRec(t, mac, opts)
	if !strings.Contains(got, "Recommendations  (ctx 8192, estimates ± expected error)\n  "+note+"\n") {
		t.Errorf("metal note missing or misplaced:\n%s", got)
	}
	if strings.Contains(renderRec(t, gpuHW(), opts), "calibrated on one M1") {
		t.Error("note must not appear for cuda")
	}
}

func TestConfigUsageLines(t *testing.T) {
	got := renderRec(t, gpuHW(), opts)
	if !strings.Contains(got, "Restart Ollama after setting these.") {
		t.Errorf("missing restart line:\n%s", got)
	}
	if !strings.Contains(got, "ollama create ") || !strings.Contains(got, "-tuned -f Modelfile") {
		t.Errorf("missing create line:\n%s", got)
	}
	cpu := renderRec(t, cpuHW(), opts)
	if strings.Contains(cpu, "Restart Ollama") {
		t.Error("no env, no restart line")
	}
	if !strings.Contains(cpu, "ollama create llama3.2-3b-instruct-q4_K_M-tuned -f Modelfile") {
		t.Errorf("create name:\n%s", cpu)
	}
	if !strings.Contains(cpu, "downloads up to ~") {
		t.Errorf("footer wording:\n%s", cpu)
	}
}
