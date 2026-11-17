package output

import (
	"fmt"
	"io"

	"github.com/shra26/ollama-compatibility-checker/internal/recommend"
)

func runtimeName(cfg recommend.Config) string {
	if cfg.Runtime == "" {
		return recommend.RuntimeOllama
	}
	return cfg.Runtime
}

// runtimeConfig formats cfg for a non-Ollama runtime; ok is false for Ollama.
func runtimeConfig(cfg recommend.Config) (recommend.RuntimeConfig, bool) {
	switch cfg.Runtime {
	case recommend.RuntimeLlamaCpp:
		return recommend.LlamaCppConfig(cfg), true
	case recommend.RuntimeLMStudio:
		return recommend.LMStudioConfig(cfg), true
	}
	return recommend.RuntimeConfig{}, false
}

func runtimeLabel(rt string) string {
	if rt == recommend.RuntimeLMStudio {
		return "LM Studio"
	}
	return "llama.cpp"
}

// renderRuntime writes the llama.cpp or LM Studio flavour of the config block.
func renderRuntime(w io.Writer, heading string, cfg recommend.Config, rc recommend.RuntimeConfig) {
	fmt.Fprintln(w, heading)
	indentBlock(w, rc.Command, "  ")
	if len(rc.Settings) > 0 {
		fmt.Fprintln(w, "  Set in the app:")
		for _, s := range rc.Settings {
			fmt.Fprintln(w, "    "+s)
		}
	}
	for _, n := range rc.Notes {
		fmt.Fprintln(w, "  note: "+n)
	}
	if cfg.CtxNote != "" {
		fmt.Fprintln(w, "  note: context "+cfg.CtxNote)
	}
}
