package output

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/shra26/ollama-compatibility-checker/internal/catalog"
	"github.com/shra26/ollama-compatibility-checker/internal/estimate"
	"github.com/shra26/ollama-compatibility-checker/internal/hw"
	"github.com/shra26/ollama-compatibility-checker/internal/recommend"
)

const (
	ansiBold   = "\x1b[1m"
	ansiYellow = "\x1b[33m"
	ansiReset  = "\x1b[0m"
)

func paint(color bool, code, s string) string {
	if !color {
		return s
	}
	return code + s + ansiReset
}

func newTab(w io.Writer) *tabwriter.Writer { return tabwriter.NewWriter(w, 0, 0, 2, ' ', 0) }

func gbStr(b uint64) string { return fmt.Sprintf("%.1f GB", float64(b)/1e9) }

func backendLabel(b string) string {
	switch b {
	case "cuda":
		return "CUDA"
	case "rocm":
		return "ROCm"
	case "metal":
		return "Metal"
	case "vulkan":
		return "Vulkan"
	case "cpu":
		return "CPU"
	}
	return b
}

// RenderScan writes the Hardware section.
func RenderScan(w io.Writer, h hw.Hardware, color bool) {
	fmt.Fprintln(w, paint(color, ansiBold, "Hardware"))
	tw := newTab(w)
	supported := false
	for i, g := range h.GPUs {
		bw := ""
		if g.BandwidthGBs > 0 {
			bw = fmt.Sprintf("%.0f GB/s", g.BandwidthGBs)
		}
		backend := backendLabel(g.Backend)
		if g.Driver != "" {
			backend += " (driver " + g.Driver + ")"
		}
		var flags []string
		if g.BandwidthLowConfidence {
			flags = append(flags, "low confidence")
		}
		if !g.Supported {
			flags = append(flags, "not used by Ollama")
		} else {
			supported = true
		}
		row := fmt.Sprintf("  GPU %d\t%s\t%s VRAM\t%s\t%s", i, g.Name, gbStr(g.VRAMBytes), bw, backend)
		if len(flags) > 0 {
			row += "\t" + strings.Join(flags, ", ")
		}
		fmt.Fprintln(tw, row)
	}
	ram := gbStr(h.RAMBytes)
	if h.RAMBandwidthGBs > 0 {
		ram += fmt.Sprintf("  ~%.0f GB/s", h.RAMBandwidthGBs)
	}
	fmt.Fprintf(tw, "  RAM\t%s\n", ram)
	cpu := h.CPU.Model
	if h.CPU.Cores > 0 {
		cpu += fmt.Sprintf("  %dC/%dT", h.CPU.Cores, h.CPU.Threads)
	}
	if h.CPU.AVX2 {
		cpu += "  AVX2"
	}
	if h.CPU.AVX512 {
		cpu += " AVX-512"
	}
	fmt.Fprintf(tw, "  CPU\t%s\n", cpu)
	tw.Flush()
	if !supported {
		fmt.Fprintln(w, "  No supported GPU; estimates use CPU and system RAM.")
	}
	for _, s := range h.Warnings {
		fmt.Fprintln(w, paint(color, ansiYellow, "  ! "+s))
	}
}

var kindLabel = map[string]string{"fastest": "Fastest", "balanced": "Balanced", "largest": "Largest"}

func label(kind string) string {
	if l, ok := kindLabel[kind]; ok {
		return l
	}
	return kind
}

func fitStr(e estimate.Estimate) string {
	s := "CPU"
	if e.GPUFraction > 0 {
		s = fmt.Sprintf("%.0f%% GPU", e.GPUFraction*100)
	}
	if e.LowConfidence {
		s += " (low confidence)"
	}
	return s
}

func estRow(tw io.Writer, first, indent string, e estimate.Estimate) {
	fmt.Fprintf(tw, "%s%s\t%s\t%s\t%s\t%s\n", indent, first, e.Variant.Tag, speedStr(e), gbStr(e.NeedBytes), fitStr(e))
}

// speedStr is the estimated tok/s with its expected error, e.g. "~21 tok/s ±15%".
func speedStr(e estimate.Estimate) string {
	return fmt.Sprintf("~%.0f tok/s ±%d%%", e.TokS, e.ErrPct)
}

// memLabel rounds a memory size to whole GiB, the way RAM and cards are sold.
func memLabel(b uint64) string {
	return fmt.Sprintf("%d GB", (b+1<<29)>>30)
}

// ConfigTarget is the estimate that Config is built for: Balanced Best, or
// Balanced Closest when unmet. Nil when there is neither.
func ConfigTarget(picks []recommend.Pick) *estimate.Estimate {
	for _, p := range picks {
		if p.Kind != "balanced" {
			continue
		}
		if p.Met {
			return p.Best
		}
		return p.Closest
	}
	return nil
}

// RenderRecommend writes Hardware, Recommendations, Config and the bench footer.
func RenderRecommend(w io.Writer, h hw.Hardware, picks []recommend.Pick, cfg recommend.Config, o recommend.Options, up *recommend.Upgrade, color bool) {
	RenderScan(w, h, color)
	fmt.Fprintln(w)
	fmt.Fprintln(w, paint(color, ansiBold, recHeader(o)))
	if usesMetal(h) {
		fmt.Fprintln(w, "  Note: Apple Silicon estimates are calibrated on one M1 8 GB; other chips may differ. Run `ollama-compat bench` to measure.")
	}
	tw := newTab(w)
	for _, p := range picks {
		switch {
		case p.Met && p.Best != nil:
			estRow(tw, label(p.Kind), "  ", *p.Best)
			for _, r := range p.RunnersUp {
				estRow(tw, "also", "      ", r)
			}
			if p.Note != "" {
				fmt.Fprintln(tw, "      "+p.Note)
			}
		default:
			fmt.Fprintf(tw, "  %s\t(no pick)\n", label(p.Kind))
			if p.Closest != nil {
				estRow(tw, "closest", "      ", *p.Closest)
			}
			if p.Note != "" {
				fmt.Fprintln(tw, "      "+p.Note)
			}
		}
	}
	tw.Flush()
	switch {
	case up != nil && up.FullGPU:
		fmt.Fprintf(w, "  With %s %s, %s would run fully on GPU (%s).\n",
			memLabel(up.MemBytes), up.MemKind, up.Largest.Variant.Tag, speedStr(up.Largest))
	case up != nil:
		fmt.Fprintf(w, "  With %s %s, Largest would be %s (%s, needs %s).\n",
			memLabel(up.MemBytes), up.MemKind, up.Largest.Variant.Tag, speedStr(up.Largest), gbStr(up.Largest.NeedBytes))
	}

	if t := ConfigTarget(picks); t != nil && cfg.PullCmd != "" {
		heading := "Config for Balanced"
		if !isMet(picks, "balanced") {
			heading += " (closest)"
		}
		fmt.Fprintln(w)
		if rc, ok := runtimeConfig(cfg); ok {
			renderRuntime(w, paint(color, ansiBold, heading+" ["+runtimeLabel(cfg.Runtime)+"]"), cfg, rc)
		} else {
			fmt.Fprintln(w, paint(color, ansiBold, heading))
			fmt.Fprintln(w, "  "+cfg.PullCmd)
			renderEnv(w, h.OS, cfg)
			fmt.Fprintln(w, "  Modelfile:")
			indentBlock(w, cfg.Modelfile, "    ")
			fmt.Fprintf(w, "  ollama create %s -f Modelfile\n", tunedName(cfg.PullCmd))
			if cfg.CtxNote != "" {
				fmt.Fprintln(w, "  note: context "+cfg.CtxNote)
			}
		}
	}

	if total, any := downloadBytes(picks); any {
		fmt.Fprintln(w)
		fmt.Fprintf(w, "Run `ollama-compat bench` to measure these (downloads up to ~%.1f GB, asks first).\n", float64(total)/1e9)
	}
}

// usesMetal reports whether the first supported GPU runs on Metal.
func usesMetal(h hw.Hardware) bool {
	for _, g := range h.GPUs {
		if g.Supported {
			return g.Backend == "metal"
		}
	}
	return false
}

// tunedName derives a model name for `ollama create` from the pull command.
func tunedName(pullCmd string) string {
	tag := strings.TrimPrefix(pullCmd, "ollama pull ")
	return strings.ReplaceAll(tag, ":", "-") + "-tuned"
}

func isMet(picks []recommend.Pick, kind string) bool {
	for _, p := range picks {
		if p.Kind == kind {
			return p.Met
		}
	}
	return false
}

func downloadBytes(picks []recommend.Pick) (uint64, bool) {
	seen := map[string]bool{}
	var total uint64
	for _, p := range picks {
		if p.Met && p.Best != nil && !seen[p.Best.Variant.Tag] {
			seen[p.Best.Variant.Tag] = true
			total += p.Best.Variant.SizeBytes
		}
	}
	return total, len(seen) > 0
}

func renderEnv(w io.Writer, goos string, cfg recommend.Config) {
	if len(cfg.Env) == 0 {
		return
	}
	type sec struct{ key, title string }
	var secs []sec
	switch goos {
	case "darwin":
		secs = []sec{{"posix", "env:"}, {"macos-app", "env (for the Ollama.app):"}}
	case "windows":
		secs = []sec{{"powershell", "env:"}, {"windows-persistent", "env (persistent):"}}
	default:
		secs = []sec{{"posix", "env:"}}
	}
	shown := false
	for _, s := range secs {
		if body := cfg.EnvSyntax[s.key]; body != "" {
			fmt.Fprintln(w, "  "+s.title)
			indentBlock(w, body, "    ")
			shown = true
		}
	}
	if shown {
		fmt.Fprintln(w, "  Restart Ollama after setting these.")
	}
}

func indentBlock(w io.Writer, s, indent string) {
	for _, line := range strings.Split(strings.TrimRight(s, "\n"), "\n") {
		fmt.Fprintln(w, indent+line)
	}
}

// RenderCatalog lists every model variant, sorted by family then size.
func RenderCatalog(w io.Writer, ms []catalog.Model) {
	type row struct {
		family string
		params float64
		tag    string
		quant  string
		size   uint64
		ctx    int
		q      int
	}
	var rows []row
	for _, m := range ms {
		for _, v := range m.Variants {
			rows = append(rows, row{m.Family, m.ParamsB, v.Tag, v.Quant, v.SizeBytes, m.MaxCtx, m.Quality})
		}
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].family != rows[j].family {
			return rows[i].family < rows[j].family
		}
		if rows[i].params != rows[j].params {
			return rows[i].params < rows[j].params
		}
		return rows[i].tag < rows[j].tag
	})
	tw := newTab(w)
	fmt.Fprintln(tw, "TAG\tPARAMS\tQUANT\tSIZE\tMAX CTX\tQUALITY")
	for _, r := range rows {
		fmt.Fprintf(tw, "%s\t%.1fB\t%s\t%s\t%d\t%d\n", r.tag, r.params, r.quant, gbStr(r.size), r.ctx, r.q)
	}
	tw.Flush()
}

// recHeader is the Recommendations title, naming the task filter when set.
func recHeader(o recommend.Options) string {
	if o.Task != "" {
		return fmt.Sprintf("Recommendations  (ctx %d, estimates ± expected error, task: %s)", o.Ctx, o.Task)
	}
	return fmt.Sprintf("Recommendations  (ctx %d, estimates ± expected error)", o.Ctx)
}
