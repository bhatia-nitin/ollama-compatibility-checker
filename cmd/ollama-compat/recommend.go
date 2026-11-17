package main

import (
	"errors"
	"fmt"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"github.com/shra26/ollama-compatibility-checker/internal/catalog"
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
	f.String("task", "", "only recommend models tagged for this task: "+strings.Join(catalog.Tasks, ", "))
	f.String("runtime", recommend.RuntimeOllama, "runtime to print config for: "+strings.Join(recommend.Runtimes, ", "))
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
	rt, _ := f.GetString("runtime")
	task, _ := f.GetString("task")
	switch {
	case minTPS < 0:
		return recommend.Options{}, errors.New("--min-tps must not be negative")
	case ctx <= 0:
		return recommend.Options{}, errors.New("--ctx must be positive")
	case top < 0:
		return recommend.Options{}, errors.New("--top must not be negative")
	case !recommend.ValidRuntime(rt):
		return recommend.Options{}, fmt.Errorf("--runtime %q is not valid; use one of: %s", rt, strings.Join(recommend.Runtimes, ", "))
	case task != "" && !catalog.ValidTask(task):
		return recommend.Options{}, fmt.Errorf("--task %q is not valid; use one of: %s", task, strings.Join(catalog.Tasks, ", "))
	}
	return recommend.Options{MinTPS: minTPS, Ctx: ctx, Top: top, Runtime: rt, Task: task}, nil
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
		up := recommend.NearMiss(h, ms, o, picks)
		var cfg recommend.Config
		if e := output.ConfigTarget(picks); e != nil {
			cfg = recommend.BuildConfig(h, *e, o)
		}
		if asJSON, _ := f.GetBool("json"); asJSON {
			doc := output.RecommendDoc(h, picks, cfg)
			doc.Task = o.Task
			doc.WithNearMiss(up)
			return output.WriteJSON(cmd.OutOrStdout(), doc)
		}
		output.RenderRecommend(cmd.OutOrStdout(), h, picks, cfg, o, up, useColor(cmd))
		return nil
	}
}
