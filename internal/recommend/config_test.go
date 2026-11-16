package recommend

import (
	"strings"
	"testing"

	"github.com/shra26/ollama-compatibility-checker/internal/catalog"
	"github.com/shra26/ollama-compatibility-checker/internal/estimate"
	"github.com/shra26/ollama-compatibility-checker/internal/hw"
)

func est(h hw.Hardware, m catalog.Model, ctx int) estimate.Estimate {
	return estimate.EstimateVariant(h, m, m.Variants[0], estimate.Params{Ctx: ctx, EffGPU: 0.65, EffCPU: 0.55})
}

func TestConfigFullFit(t *testing.T) {
	h := cudaHW()
	m := cm("mid", 5_000_000_000, 70)
	c := BuildConfig(h, est(h, m, 8192), Options{Ctx: 8192})
	if c.NumGPU != 99 || c.NumCtx != 8192 {
		t.Fatalf("%+v", c)
	}
	if c.Env["OLLAMA_FLASH_ATTENTION"] != "1" || c.Env["OLLAMA_KV_CACHE_TYPE"] != "q8_0" {
		t.Fatalf("env %v", c.Env)
	}
	want := "FROM mid:q4\nPARAMETER num_ctx 8192\nPARAMETER num_gpu 99\n"
	if c.Modelfile != want {
		t.Fatalf("modelfile %q", c.Modelfile)
	}
	if c.PullCmd != "ollama pull mid:q4" {
		t.Fatalf("pull %q", c.PullCmd)
	}
}

func TestConfigCPUOnlyNoEnv(t *testing.T) {
	h := hw.Hardware{RAMBytes: 32 * gib, RAMBandwidthGBs: 50}
	c := BuildConfig(h, est(h, cm("mid", 5_000_000_000, 70), 8192), Options{Ctx: 8192})
	if len(c.Env) != 0 || len(c.EnvSyntax) != 0 || c.NumGPU != 0 {
		t.Fatalf("%+v", c)
	}
}

func TestConfigCtxCapped(t *testing.T) {
	h := cudaHW()
	m := cm("mid", 5_000_000_000, 70)
	m.MaxCtx = 32768
	c := BuildConfig(h, est(h, m, 65536), Options{Ctx: 65536})
	if c.NumCtx != 32768 || !strings.Contains(c.CtxNote, "32768") {
		t.Fatalf("%+v", c)
	}
}

func heavyKV(size uint64) catalog.Model {
	m := cm("kv", size, 70)
	m.KVHeads = 32 // q8 KV: 2 GiB at 8192, 1 GiB at 4096
	return m
}

func TestConfigCtxShrinksToFit(t *testing.T) {
	h := cudaHW()
	m := heavyKV(9_200_000_000)
	e := est(h, m, 8192)
	if e.GPUFraction == 1 {
		t.Fatal("test setup: should not fit at 8192")
	}
	c := BuildConfig(h, e, Options{Ctx: 8192})
	if c.NumCtx != 4096 || c.NumGPU != 99 || !strings.Contains(c.CtxNote, "reduced from 8192 to 4096") {
		t.Fatalf("%+v", c)
	}
	if !strings.Contains(c.Modelfile, "num_ctx 4096") {
		t.Fatalf("modelfile %q", c.Modelfile)
	}
}

func TestConfigCtxNoFitKeepsRequested(t *testing.T) {
	h := cudaHW()
	m := heavyKV(11_000_000_000)
	e := est(h, m, 8192)
	c := BuildConfig(h, e, Options{Ctx: 8192})
	if c.NumCtx != 8192 || c.NumGPU != e.NumGPU || c.NumGPU == 99 {
		t.Fatalf("%+v", c)
	}
}

func TestEnvSyntax(t *testing.T) {
	h := cudaHW()
	c := BuildConfig(h, est(h, cm("mid", 5_000_000_000, 70), 8192), Options{Ctx: 8192})
	checks := map[string][]string{
		"posix":              {"export OLLAMA_FLASH_ATTENTION=1", "export OLLAMA_KV_CACHE_TYPE=q8_0"},
		"powershell":         {`$env:OLLAMA_FLASH_ATTENTION="1"`, `$env:OLLAMA_KV_CACHE_TYPE="q8_0"`},
		"windows-persistent": {"setx OLLAMA_FLASH_ATTENTION 1", "setx OLLAMA_KV_CACHE_TYPE q8_0"},
		"macos-app":          {"launchctl setenv OLLAMA_FLASH_ATTENTION 1", "launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0"},
	}
	for k, subs := range checks {
		s, ok := c.EnvSyntax[k]
		if !ok {
			t.Fatalf("missing %s", k)
		}
		for _, sub := range subs {
			if !strings.Contains(s, sub) {
				t.Fatalf("%s missing %q in %q", k, sub, s)
			}
		}
		if !strings.Contains(s, "applies to all models") {
			t.Fatalf("%s missing KV note", k)
		}
		if strings.Index(s, "FLASH") > strings.Index(s, "KV_CACHE") {
			t.Fatalf("%s not sorted", k)
		}
	}
}

func TestNoKVEnvWhenKVQuantNotOK(t *testing.T) {
	h := cudaHW()
	m := cm("mid", 5_000_000_000, 70)
	m.KVQuantOK = false
	c := BuildConfig(h, est(h, m, 8192), Options{Ctx: 8192})
	if _, ok := c.Env["OLLAMA_KV_CACHE_TYPE"]; ok || c.Env["OLLAMA_FLASH_ATTENTION"] != "1" {
		t.Fatalf("env %v", c.Env)
	}
	if strings.Contains(c.EnvSyntax["posix"], "applies to all models") {
		t.Fatal("KV note without KV var")
	}
}
