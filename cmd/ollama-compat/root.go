package main

import (
	"io"

	"github.com/spf13/cobra"
)

// version is set at build time via -ldflags "-X main.version=...".
var version = "dev"

// newRootCmd builds the root command. Subcommands are attached with AddCommand.
func newRootCmd(stdout, stderr io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "ollama-compat",
		Short:   "Recommend Ollama models that fit your hardware",
		Version: version,
	}
	cmd.SetOut(stdout)
	cmd.SetErr(stderr)
	cmd.SetVersionTemplate("ollama-compat {{.Version}}\n")

	pf := cmd.PersistentFlags()
	pf.Bool("json", false, "emit machine-readable JSON")
	pf.String("catalog", "", "path to a custom model catalog file")
	pf.Bool("no-color", false, "disable colored output")

	return cmd
}
