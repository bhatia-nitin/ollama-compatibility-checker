package output

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/shra26/ollama-compatibility-checker/internal/recommend"
)

func TestRenderRuntimes(t *testing.T) {
	tests := []struct {
		runtime string
		want    []string
		not     []string
	}{
		{recommend.RuntimeLlamaCpp, []string{"Config for Balanced [llama.cpp]", "llama-server", "-ngl 99", "-fa on", "-ctk q8_0", "-ctv q8_0", "/path/to/", ".gguf", "any GGUF of"}, []string{"Modelfile", "ollama pull", "OLLAMA_"}},
		{recommend.RuntimeLMStudio, []string{"Config for Balanced [LM Studio]", "lms load /path/to/", "--context-length 8192", "--gpu max", "Set in the app:", "Flash Attention: on", "KV cache quantization: q8_0"}, []string{"Modelfile", "ollama pull", "OLLAMA_"}},
		{recommend.RuntimeOllama, []string{"Config for Balanced\n", "ollama pull", "Modelfile"}, []string{"llama-server", "lms load"}},
	}
	for _, tt := range tests {
		t.Run(tt.runtime, func(t *testing.T) {
			o := opts
			o.Runtime = tt.runtime
			got := renderRec(t, gpuHW(), o)
			for _, w := range tt.want {
				if !strings.Contains(got, w) {
					t.Errorf("missing %q in:\n%s", w, got)
				}
			}
			for _, n := range tt.not {
				if strings.Contains(got, n) {
					t.Errorf("unexpected %q in:\n%s", n, got)
				}
			}
		})
	}
}

func TestJSONRuntime(t *testing.T) {
	h := gpuHW()
	for rt, wantCmd := range map[string]string{"ollama": "", "llamacpp": "llama-server", "lmstudio": "lms load"} {
		o := opts
		o.Runtime = rt
		picks := recommend.Recommend(h, testCatalog(t), o)
		cfg := recommend.BuildConfig(h, *ConfigTarget(picks), o)
		var b bytes.Buffer
		if err := WriteJSON(&b, RecommendDoc(h, picks, cfg)); err != nil {
			t.Fatal(err)
		}
		var d struct {
			Config map[string]any `json:"config"`
		}
		if err := json.Unmarshal(b.Bytes(), &d); err != nil {
			t.Fatal(err)
		}
		if d.Config["runtime"] != rt {
			t.Errorf("%s: runtime = %v", rt, d.Config["runtime"])
		}
		cmd, has := d.Config["runtime_command"].(string)
		if wantCmd == "" && has {
			t.Errorf("ollama must not carry runtime_command")
		}
		if wantCmd != "" && !strings.Contains(cmd, wantCmd) {
			t.Errorf("%s: runtime_command = %q", rt, cmd)
		}
		if d.Config["pull_cmd"] == nil || d.Config["modelfile"] == nil {
			t.Errorf("%s: existing fields must remain", rt)
		}
	}
}
