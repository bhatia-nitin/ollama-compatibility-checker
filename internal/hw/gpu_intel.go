package hw

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"strings"
)

// lspciQuoted splits an `lspci -mm` line into its quoted fields.
var lspciQuoted = regexp.MustCompile(`"([^"]*)"`)

// ParseLspciIntel returns Intel display controllers found in `lspci -mm`
// output. Ollama has no supported backend for them, so they are unsupported.
func ParseLspciIntel(out []byte) []GPU {
	var gpus []GPU
	for _, line := range strings.Split(string(out), "\n") {
		q := lspciQuoted.FindAllStringSubmatch(line, -1)
		if len(q) < 3 || !strings.HasPrefix(q[0][1], "VGA") || !strings.Contains(q[1][1], "Intel") {
			continue
		}
		gpus = append(gpus, GPU{
			Vendor:  "intel",
			Name:    strings.TrimSpace(q[1][1] + " " + q[2][1]),
			Backend: "cpu",
		})
	}
	return gpus
}

// DetectIntel runs lspci. A missing lspci yields nothing and no warning.
func DetectIntel(ctx context.Context, run Runner) ([]GPU, []string) {
	out, err := run(ctx, "lspci", "-mm")
	if err != nil {
		if errors.Is(err, exec.ErrNotFound) {
			return nil, nil
		}
		return nil, []string{fmt.Sprintf("lspci failed (%v)", err)}
	}
	return ParseLspciIntel(out), nil
}
