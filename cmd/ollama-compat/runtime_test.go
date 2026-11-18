package main

import (
	"strings"
	"testing"
)

func TestRuntimeFlag(t *testing.T) {
	const cat = "testdata/catalog.json"
	for _, args := range [][]string{{"--runtime", "vllm"}, {"recommend", "--runtime", "vllm"}} {
		_, _, err := run(t, args...)
		if err == nil || !strings.Contains(err.Error(), "ollama, llamacpp, lmstudio") {
			t.Errorf("%v: want error listing valid values, got %v", args, err)
		}
	}
	for _, rt := range []string{"ollama", "llamacpp", "lmstudio"} {
		for _, args := range [][]string{{"--runtime", rt, "--catalog", cat}, {"recommend", "--runtime", rt, "--catalog", cat}} {
			if _, _, err := run(t, args...); err != nil {
				t.Errorf("%v: %v", args, err)
			}
		}
	}
	out, _, _ := run(t, "--runtime", "llamacpp", "--catalog", cat)
	if !strings.Contains(out, "llama-server") || strings.Contains(out, "Modelfile") {
		t.Errorf("llamacpp output:\n%s", out)
	}
}
