package main

import (
	"fmt"
	"io"
	"os"

	"github.com/spf13/cobra"
	"golang.org/x/term"

	"github.com/shra26/ollama-compatibility-checker/internal/catalog"
)

// loadCatalog returns the embedded catalog, or the validated file at path.
func loadCatalog(path string) ([]catalog.Model, error) {
	if path == "" {
		return catalog.Embedded(), nil
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open catalog: %w", err)
	}
	defer f.Close()
	ms, err := catalog.Load(f)
	if err != nil {
		return nil, fmt.Errorf("load catalog %s: %w", path, err)
	}
	if err := catalog.Validate(ms); err != nil {
		return nil, fmt.Errorf("invalid catalog %s: %w", path, err)
	}
	return ms, nil
}

// useColor reports whether ANSI color is allowed: stdout is a terminal and
// neither --no-color nor a non-empty NO_COLOR is set.
func useColor(cmd *cobra.Command) bool {
	if off, _ := cmd.Flags().GetBool("no-color"); off {
		return false
	}
	if os.Getenv("NO_COLOR") != "" {
		return false
	}
	return isTerminal(cmd.OutOrStdout())
}

func isTerminal(w io.Writer) bool {
	f, ok := w.(*os.File)
	return ok && term.IsTerminal(int(f.Fd()))
}
