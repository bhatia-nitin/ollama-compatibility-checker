package recommend

import (
	"fmt"
	"strings"
)

// Supported runtimes for the --runtime flag.
const (
	RuntimeOllama   = "ollama"
	RuntimeLlamaCpp = "llamacpp"
	RuntimeLMStudio = "lmstudio"
)

// Runtimes lists the valid --runtime values.
var Runtimes = []string{RuntimeOllama, RuntimeLlamaCpp, RuntimeLMStudio}

// ValidRuntime reports whether r is a supported runtime name.
func ValidRuntime(r string) bool {
	for _, v := range Runtimes {
		if v == r {
			return true
		}
	}
	return false
}

// RuntimeConfig is the non-Ollama rendering of a Config.
type RuntimeConfig struct {
	Command  string   // shell command to run
	Settings []string // settings to apply by hand in the app (LM Studio)
	Notes    []string // one-line remarks
}

// ggufPlaceholder builds a clearly marked placeholder model path from the
// catalog name and quant, e.g. /path/to/Qwen2.5-7B-Q4_K_M.gguf.
func ggufPlaceholder(c Config) string {
	name := strings.ReplaceAll(strings.TrimSpace(c.ModelName), " ", "-")
	return "/path/to/" + name + "-" + strings.ToUpper(c.Quant) + ".gguf"
}

func ggufNote(c Config) string {
	return fmt.Sprintf("replace the placeholder path; any GGUF of %s at %s works.", c.ModelName, strings.ToUpper(c.Quant))
}

// LlamaCppConfig formats c as a llama-server command.
func LlamaCppConfig(c Config) RuntimeConfig {
	args := []string{
		"llama-server",
		"-m " + ggufPlaceholder(c),
		fmt.Sprintf("-c %d", c.NumCtx),
		fmt.Sprintf("-ngl %d", c.NumGPU),
	}
	if c.Env["OLLAMA_FLASH_ATTENTION"] == "1" {
		args = append(args, "-fa on")
	}
	if c.Env["OLLAMA_KV_CACHE_TYPE"] == "q8_0" {
		args = append(args, "-ctk q8_0", "-ctv q8_0")
	}
	return RuntimeConfig{
		Command: strings.Join(args, " \\\n  "),
		Notes:   []string{ggufNote(c)},
	}
}

// LMStudioConfig formats c as an `lms load` command plus app-only settings.
func LMStudioConfig(c Config) RuntimeConfig {
	gpu := "max"
	switch {
	case c.NumGPU <= 0:
		gpu = "off"
	case c.NumGPU < 99 && c.Layers > 0 && c.NumGPU < c.Layers:
		gpu = fmt.Sprintf("%.2f", float64(c.NumGPU)/float64(c.Layers))
	}
	rc := RuntimeConfig{
		Command: fmt.Sprintf("lms load %s --context-length %d --gpu %s", ggufPlaceholder(c), c.NumCtx, gpu),
		Notes:   []string{ggufNote(c)},
	}
	if c.Env["OLLAMA_FLASH_ATTENTION"] == "1" {
		rc.Settings = append(rc.Settings, "Flash Attention: on")
	}
	if c.Env["OLLAMA_KV_CACHE_TYPE"] == "q8_0" {
		rc.Settings = append(rc.Settings, "KV cache quantization: q8_0 for K and V")
	}
	if len(rc.Settings) > 0 {
		rc.Notes = append(rc.Notes, "lms load has no flag for these; set them in the model's load settings in the LM Studio app.")
	}
	return rc
}
