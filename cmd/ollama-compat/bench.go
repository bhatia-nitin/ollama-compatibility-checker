package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"time"

	"github.com/spf13/cobra"
	"golang.org/x/term"

	"github.com/shra26/ollama-compatibility-checker/internal/bench"
	"github.com/shra26/ollama-compatibility-checker/internal/estimate"
	"github.com/shra26/ollama-compatibility-checker/internal/output"
	"github.com/shra26/ollama-compatibility-checker/internal/recommend"
)

// benchEnv holds the pieces of bench that touch the outside world, so tests
// can point it at an httptest server.
type benchEnv struct {
	// client returns the Ollama client and whether its host is loopback.
	client func() (*bench.Client, bool)
	// isTTY reports whether stdin is interactive.
	isTTY func() bool
}

func hostBenchEnv() benchEnv {
	return benchEnv{
		client: func() (*bench.Client, bool) {
			url, loop := bench.HostFromEnv()
			return &bench.Client{BaseURL: url}, loop
		},
		isTTY: func() bool { return term.IsTerminal(int(os.Stdin.Fd())) },
	}
}

// shortlist is the Best of each met pick, deduplicated by tag.
func shortlist(picks []recommend.Pick) []estimate.Estimate {
	var out []estimate.Estimate
	seen := map[string]bool{}
	for _, p := range picks {
		if !p.Met || p.Best == nil || seen[p.Best.Variant.Tag] {
			continue
		}
		seen[p.Best.Variant.Tag] = true
		out = append(out, *p.Best)
	}
	return out
}

func newBenchCmd(detect detectFunc, env benchEnv) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "bench",
		Short: "Benchmark the shortlist on a local Ollama (opt-in, may download models)",
		Long: "Measures real tokens per second for the recommended models through a local Ollama server.\n" +
			"This is the only command that downloads data. It shows the total size and asks first.",
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			f := cmd.Flags()
			ro, err := recommendOptions(f)
			if err != nil {
				return err
			}
			runs, _ := f.GetInt("runs")
			if runs <= 0 {
				return errors.New("--runs must be positive")
			}
			var maxDL uint64
			if s, _ := f.GetString("max-download"); s != "" {
				if maxDL, err = bench.ParseSize(s); err != nil {
					return fmt.Errorf("--max-download: %w", err)
				}
			}
			path, _ := f.GetString("catalog")
			ms, err := loadCatalog(path)
			if err != nil {
				return err
			}
			base := cmd.Context()
			if base == nil {
				base = context.Background()
			}
			ctx, stop := signal.NotifyContext(base, os.Interrupt)
			defer stop()

			h := detect(ctx)
			sl := shortlist(recommend.Recommend(h, ms, ro))
			if len(sl) == 0 {
				return errors.New("no model fits this machine, so there is nothing to benchmark")
			}
			client, loopback := env.client()
			o := bench.Options{Ctx: ro.Ctx, Runs: runs, MaxDownload: maxDL, Loopback: loopback,
				In: cmd.InOrStdin(), Out: cmd.ErrOrStderr(), IsTTY: env.isTTY()}
			o.Yes, _ = f.GetBool("yes")
			o.InstalledOnly, _ = f.GetBool("installed-only")
			o.Keep, _ = f.GetBool("keep")
			o.AllowRemote, _ = f.GetBool("allow-remote")

			results, err := bench.Run(ctx, client, sl, o)
			if err != nil {
				return err
			}
			if results == nil {
				return nil // declined at the prompt
			}
			if asJSON, _ := f.GetBool("json"); asJSON {
				err = output.WriteJSON(cmd.OutOrStdout(), output.BenchDoc(h, results))
			} else {
				output.RenderBench(cmd.OutOrStdout(), results)
			}
			if err != nil {
				return err
			}
			if share, _ := f.GetString("share"); share != "" {
				ov, _ := client.Version(ctx)
				return writeShare(cmd.ErrOrStderr(), share, calibrationRecord(h, results, ov, time.Now()))
			}
			return nil
		},
	}
	addRecommendFlags(cmd)
	f := cmd.Flags()
	f.Bool("yes", false, "skip the download confirmation (for scripts)")
	f.String("max-download", "", "refuse if the total download exceeds this size, for example 20GB")
	f.Bool("installed-only", false, "benchmark only models already installed; downloads nothing")
	f.Bool("keep", false, "keep models downloaded for the benchmark")
	f.Bool("allow-remote", false, "allow benchmarking a non-local OLLAMA_HOST")
	f.Int("runs", 3, "timed runs per model (the median is reported)")
	f.String("share", "", "save a calibration record to this file for contributing (nothing is sent)")
	f.Lookup("share").NoOptDefVal = "ollama-compat-calibration.json"
	return cmd
}
