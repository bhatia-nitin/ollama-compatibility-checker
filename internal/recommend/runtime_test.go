package recommend

import (
	"strings"
	"testing"
)

func withFields(c Config, ngl int, env map[string]string) Config {
	c.NumGPU, c.Env = ngl, env
	return c
}

func TestLlamaCppConfig(t *testing.T) {
	full := map[string]string{"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"}
	base := Config{ModelName: "Qwen2.5 7B", Quant: "q4_K_M", NumCtx: 8192, Layers: 28}
	tests := []struct {
		name string
		c    Config
		want []string
		not  []string
	}{
		{"full fit", withFields(base, 99, full), []string{"llama-server \\\n", "-m /path/to/Qwen2.5-7B-Q4_K_M.gguf", "-c 8192", "-ngl 99", "-fa on", "-ctk q8_0", "-ctv q8_0"}, nil},
		{"partial", withFields(base, 17, full), []string{"-ngl 17"}, nil},
		{"cpu only", withFields(base, 0, nil), []string{"-ngl 0"}, []string{"-fa", "-ctk", "-ctv"}},
		{"flash only", withFields(base, 99, map[string]string{"OLLAMA_FLASH_ATTENTION": "1"}), []string{"-fa on"}, []string{"-ctk"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rc := LlamaCppConfig(tt.c)
			for _, w := range tt.want {
				if !strings.Contains(rc.Command, w) {
					t.Errorf("missing %q in\n%s", w, rc.Command)
				}
			}
			for _, n := range tt.not {
				if strings.Contains(rc.Command, n) {
					t.Errorf("unexpected %q in\n%s", n, rc.Command)
				}
			}
			if len(rc.Notes) != 1 || !strings.Contains(rc.Notes[0], "any GGUF of Qwen2.5 7B at Q4_K_M") {
				t.Errorf("notes %v", rc.Notes)
			}
		})
	}
}

func TestLMStudioConfig(t *testing.T) {
	full := map[string]string{"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"}
	base := Config{ModelName: "Qwen2.5 7B", Quant: "q8_0", NumCtx: 4096, Layers: 28}
	tests := []struct {
		name         string
		c            Config
		cmd          string
		wantSettings int
	}{
		{"full fit", withFields(base, 99, full), "lms load /path/to/Qwen2.5-7B-Q8_0.gguf --context-length 4096 --gpu max", 2},
		{"partial", withFields(base, 14, full), "lms load /path/to/Qwen2.5-7B-Q8_0.gguf --context-length 4096 --gpu 0.50", 2},
		{"cpu only", withFields(base, 0, nil), "lms load /path/to/Qwen2.5-7B-Q8_0.gguf --context-length 4096 --gpu off", 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rc := LMStudioConfig(tt.c)
			if rc.Command != tt.cmd {
				t.Errorf("command %q, want %q", rc.Command, tt.cmd)
			}
			if len(rc.Settings) != tt.wantSettings {
				t.Errorf("settings %v", rc.Settings)
			}
		})
	}
}

func TestValidRuntime(t *testing.T) {
	for _, r := range []string{"ollama", "llamacpp", "lmstudio"} {
		if !ValidRuntime(r) {
			t.Errorf("%s should be valid", r)
		}
	}
	for _, r := range []string{"", "vllm", "Ollama"} {
		if ValidRuntime(r) {
			t.Errorf("%q should be invalid", r)
		}
	}
}
