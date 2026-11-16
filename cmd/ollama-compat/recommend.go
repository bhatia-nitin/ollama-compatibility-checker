package main

import (
	"errors"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"github.com/shra26/ollama-compatibility-checker/internal/output"
	"github.com/shra26/ollama-compatibility-checker/internal/recommend"
)

// addRecommendFlags registers the recommend flags on cmd. Both the root
// command and `recommend` carry them.
func addRecommendFlags(cmd *cobra.Command) {
	f := cmd.Flags()
	f.Float64("min-tps", 20, "Balanced pick must reach this estimated tok/s")
	f.Int("ctx", 8192, "context length to size the KV cache for")
	f.Int("top", 3, "how many alternatives to list per pick")
}

func newRecommendCmd(detect detectFunc) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "recommend",
		Short: "Scan hardware and recommend models (default command)",
		Args:  cobra.NoArgs,
		RunE:  recommendRunE(detect),
	}
	addRecommendFlags(cmd)
	return cmd
}

// recommendOptions reads and validates the recommend flags.
func recommendOptions(f *pflag.FlagSet) (recommend.Options, error) {
	minTPS, _ := f.GetFloat64("min-tps")
	ctx, _ := f.GetInt("ctx")
	top, _ := f.GetInt("top")
	switch {
	case minTPS < 0:
		return recommend.Options{}, errors.New("--min-tps must not be negative")
	case ctx <= 0:
		return recommend.Options{}, errors.New("--ctx must be positive")
	case top < 0:
		return recommend.Options{}, errors.New("--top must not be negative")
	}
	return recommend.Options{MinTPS: minTPS, Ctx: ctx, Top: top}, nil
}

func recommendRunE(detect detectFunc) func(*cobra.Command, []string) error {
	return func(cmd *cobra.Command, _ []string) error {
		f := cmd.Flags()
		o, err := recommendOptions(f)
		if err != nil {
			return err
		}
		path, _ := f.GetString("catalog")
		ms, err := loadCatalog(path)
		if err != nil {
			return err
		}
		h := detect(cmd.Context())
		picks := recommend.Recommend(h, ms, o)
		var cfg recommend.Config
		if e := output.ConfigTarget(picks); e != nil {
			cfg = recommend.BuildConfig(h, *e, o)
		}
		if asJSON, _ := f.GetBool("json"); asJSON {
			return output.WriteJSON(cmd.OutOrStdout(), output.RecommendDoc(h, picks, cfg))
		}
		output.RenderRecommend(cmd.OutOrStdout(), h, picks, cfg, o, useColor(cmd))
		return nil
	}
}
