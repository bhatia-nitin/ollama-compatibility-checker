package bench

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/shra26/ollama-compatibility-checker/internal/estimate"
)

// BenchPrompt is the fixed prompt for every timed run (about 60 tokens).
const BenchPrompt = "Explain in a short paragraph how a hash table works. Cover how keys are turned into bucket positions, what happens when two keys land in the same bucket, and why lookups take constant time on average. Write for a developer who has never used one."

const (
	warmupPredict = 16
	runPredict    = 256
	defaultRuns   = 3
	mismatchDelta = 0.10
	skippedMsg    = "skipped (not installed)"
	downloadURL   = "https://ollama.com/download"
)

// Options controls a benchmark run.
type Options struct {
	Yes, InstalledOnly, Keep, AllowRemote bool
	// Loopback reports whether the Ollama host is this machine.
	Loopback bool
	// MaxDownload caps the bytes to pull (0 = no cap).
	MaxDownload uint64
	Ctx, Runs   int // Runs defaults to 3
	In          io.Reader
	Out         io.Writer
	IsTTY       bool
}

// Result is the measured outcome for one model.
type Result struct {
	Tag                 string
	Est                 estimate.Estimate
	DecodeTokS          float64
	PrefillTokS         float64
	MeasuredGPUFraction float64
	Mismatch            bool
	Pulled              bool
	Err                 string
}

// ParseSize parses "20GB", "1.5GiB" or a bare byte count. GB-style units are
// powers of 1000, GiB-style units powers of 1024.
func ParseSize(s string) (uint64, error) {
	t := strings.ToLower(strings.TrimSpace(s))
	i := 0
	for i < len(t) && (t[i] >= '0' && t[i] <= '9' || t[i] == '.') {
		i++
	}
	num, unit := t[:i], strings.TrimSpace(t[i:])
	if num == "" {
		return 0, fmt.Errorf("invalid size %q", s)
	}
	v, err := strconv.ParseFloat(num, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid size %q", s)
	}
	mult, ok := map[string]float64{
		"": 1, "b": 1,
		"kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12,
		"kib": 1 << 10, "mib": 1 << 20, "gib": 1 << 30, "tib": 1 << 40,
	}[unit]
	if !ok {
		return 0, fmt.Errorf("invalid size %q: unknown unit %q", s, unit)
	}
	b := math.Round(v * mult)
	if b >= math.MaxUint64 {
		return 0, fmt.Errorf("size %q is too large", s)
	}
	return uint64(b), nil
}

func gbStr(b uint64) string { return fmt.Sprintf("%.1f GB", float64(b)/1e9) }

// sameTag treats "name" and "name:latest" as equal.
func sameTag(a, b string) bool {
	return strings.TrimSuffix(a, ":latest") == strings.TrimSuffix(b, ":latest")
}

func isInstalled(installed []string, tag string) bool {
	for _, n := range installed {
		if sameTag(n, tag) {
			return true
		}
	}
	return false
}

func median(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	if n := len(s); n%2 == 0 {
		return (s[n/2-1] + s[n/2]) / 2
	}
	return s[len(s)/2]
}

// Run benchmarks the shortlist against the Ollama server behind c. No model is
// pulled unless the user saw the total download size and agreed (or Yes).
func Run(ctx context.Context, c *Client, shortlist []estimate.Estimate, o Options) ([]Result, error) {
	out := o.Out
	if out == nil {
		out = io.Discard
	}
	var in *bufio.Reader
	if o.In != nil {
		in = bufio.NewReader(o.In)
	}
	runs := o.Runs
	if runs <= 0 {
		runs = defaultRuns
	}

	// 1. Remote guard, before any request.
	if !o.Loopback {
		if !o.AllowRemote {
			return nil, errors.New("OLLAMA_HOST is not a local address; pass --allow-remote to benchmark a remote Ollama")
		}
		fmt.Fprintln(out, "Warning: OLLAMA_HOST is not local. Measurements describe the remote machine, not the one that was scanned.")
	}

	// 2-3. Reachability and installed models.
	if _, err := c.Version(ctx); err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, fmt.Errorf("cannot reach Ollama at %s (%v). Install it from %s and start it", c.BaseURL, err, downloadURL)
	}
	installed, err := c.Tags(ctx)
	if err != nil {
		return nil, fmt.Errorf("list installed models: %w", err)
	}

	// 4. Split into installed and missing.
	var avail, missing []estimate.Estimate
	var skipped []estimate.Estimate
	var missingBytes uint64
	for _, e := range shortlist {
		switch {
		case isInstalled(installed, e.Variant.Tag):
			avail = append(avail, e)
		case o.InstalledOnly:
			skipped = append(skipped, e)
		default:
			missing = append(missing, e)
			missingBytes += e.Variant.SizeBytes
		}
	}

	// 5. Installed-only: list the skipped models.
	for _, e := range skipped {
		fmt.Fprintf(out, "%s: %s\n", e.Variant.Tag, skippedMsg)
	}

	// 6. Download cap.
	if o.MaxDownload > 0 && missingBytes > o.MaxDownload {
		return nil, fmt.Errorf("download of %s exceeds --max-download %s", gbStr(missingBytes), gbStr(o.MaxDownload))
	}

	// 7. Consent.
	if len(missing) > 0 {
		fmt.Fprintf(out, "%-36s %12s  %s\n", "Model", "Size", "Installed")
		for _, e := range shortlist {
			yes := "yes"
			if !isInstalled(installed, e.Variant.Tag) {
				yes = "no"
			}
			size := gbStr(e.Variant.SizeBytes)
			if e.Variant.SizeBytes == 0 {
				size = "unknown size"
			}
			fmt.Fprintf(out, "%-36s %12s  %s\n", e.Variant.Tag, size, yes)
		}
		fmt.Fprintf(out, "Total to download: %s\n", gbStr(missingBytes))
		if !o.Yes {
			if !o.IsTTY {
				return nil, errors.New("refusing to download without --yes (stdin is not a terminal)")
			}
			fmt.Fprintf(out, "Download %s and benchmark %d models? [y/N] ", gbStr(missingBytes), len(avail)+len(missing))
			ans, err := readAnswer(ctx, in)
			if err != nil {
				return nil, err
			}
			if ans != "y" && ans != "yes" {
				fmt.Fprintln(out, "Cancelled.")
				return nil, nil
			}
		}
	}

	var results []Result
	var pulled []estimate.Estimate
	pulledSet := map[string]bool{}
	cancel := func() ([]Result, error) {
		listPulled(out, "Pulled during this run:", pulled)
		return results, ctx.Err()
	}

	// 8. Pull sequentially.
	for _, e := range missing {
		tag := e.Variant.Tag
		pr := newProgress(out, tag, e.Variant.SizeBytes, o.IsTTY)
		err := c.Pull(ctx, tag, pr.update)
		pr.finish(err == nil)
		if ctx.Err() != nil {
			return cancel()
		}
		if err != nil {
			results = append(results, Result{Tag: tag, Est: e, Err: "pull failed: " + err.Error()})
			continue
		}
		pulled = append(pulled, e)
		pulledSet[tag] = true
		avail = append(avail, e)
	}

	// 9. Benchmark each available model, in shortlist order.
	for _, e := range shortlist {
		tag := e.Variant.Tag
		if !pulledSet[tag] && !isInstalled(installed, tag) {
			if o.InstalledOnly {
				results = append(results, Result{Tag: tag, Est: e, Err: skippedMsg})
			}
			continue
		}
		fmt.Fprintf(out, "Benchmarking %s ...\n", tag)
		r := benchOne(ctx, c, e, o.Ctx, runs, pulledSet[tag])
		if ctx.Err() != nil {
			return cancel()
		}
		results = append(results, r)
	}
	results = orderLike(results, shortlist)

	// 10. Cleanup.
	if !o.Keep && len(pulled) > 0 {
		if err := cleanup(ctx, c, out, in, o, pulled); err != nil {
			return cancel()
		}
	}
	return results, nil
}

// orderLike returns res sorted by position of each tag in shortlist.
func orderLike(res []Result, shortlist []estimate.Estimate) []Result {
	pos := map[string]int{}
	for i, e := range shortlist {
		pos[e.Variant.Tag] = i
	}
	sort.SliceStable(res, func(i, j int) bool { return pos[res[i].Tag] < pos[res[j].Tag] })
	return res
}

// readAnswer reads one line, returning early with ctx.Err() on cancellation.
// A read still blocked on stdin is abandoned; the process is about to exit.
func readAnswer(ctx context.Context, in *bufio.Reader) (string, error) {
	if in == nil {
		return "", ctx.Err()
	}
	ch := make(chan string, 1)
	go func() {
		line, _ := in.ReadString('\n') // EOF with partial text still counts
		ch <- strings.ToLower(strings.TrimSpace(line))
	}()
	select {
	case ans := <-ch:
		return ans, ctx.Err()
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func listPulled(out io.Writer, header string, pulled []estimate.Estimate) {
	if len(pulled) == 0 {
		return
	}
	fmt.Fprintln(out, header)
	for _, e := range pulled {
		fmt.Fprintf(out, "  ollama rm %s\n", e.Variant.Tag)
	}
}

func cleanup(ctx context.Context, c *Client, out io.Writer, in *bufio.Reader, o Options, pulled []estimate.Estimate) error {
	var total uint64
	for _, e := range pulled {
		total += e.Variant.SizeBytes
	}
	if !o.IsTTY || o.Yes {
		listPulled(out, "Models downloaded for this benchmark were kept. Remove them with:", pulled)
		return nil
	}
	fmt.Fprintf(out, "Remove the %d models downloaded for this benchmark to free %s? [Y/n] ", len(pulled), gbStr(total))
	ans, err := readAnswer(ctx, in)
	if err != nil {
		return err
	}
	switch ans {
	case "", "y", "yes":
		for _, e := range pulled {
			if err := c.Delete(ctx, e.Variant.Tag); err != nil {
				if ctx.Err() != nil {
					return ctx.Err()
				}
				fmt.Fprintf(out, "Could not remove %s: %v\n", e.Variant.Tag, err)
				continue
			}
			fmt.Fprintf(out, "Removed %s\n", e.Variant.Tag)
		}
	default:
		listPulled(out, "Kept. Remove them later with:", pulled)
	}
	return nil
}

// benchOne measures one installed model. Errors are recorded in the Result.
func benchOne(ctx context.Context, c *Client, e estimate.Estimate, numCtx, runs int, pulled bool) Result {
	tag := e.Variant.Tag
	r := Result{Tag: tag, Est: e, Pulled: pulled}
	if _, err := c.Generate(ctx, tag, BenchPrompt, numCtx, warmupPredict); err != nil {
		r.Err = err.Error()
		return r
	}
	if ps, err := c.PS(ctx); err == nil {
		for _, m := range ps {
			if m.Name == tag && m.Size > 0 {
				r.MeasuredGPUFraction = float64(m.SizeVRAM) / float64(m.Size)
				break
			}
		}
	}
	r.Mismatch = math.Abs(r.MeasuredGPUFraction-e.GPUFraction) > mismatchDelta
	var decode, prefill []float64
	for i := 0; i < runs; i++ {
		g, err := c.Generate(ctx, tag, BenchPrompt, numCtx, runPredict)
		if err != nil {
			r.Err = err.Error()
			return r
		}
		if g.EvalDuration > 0 {
			decode = append(decode, float64(g.EvalCount)/g.EvalDuration.Seconds())
		}
		if g.PromptEvalDuration > 0 {
			prefill = append(prefill, float64(g.PromptEvalCount)/g.PromptEvalDuration.Seconds())
		}
	}
	if len(decode) == 0 {
		r.Err = "no timing data returned"
		return r
	}
	r.DecodeTokS, r.PrefillTokS = median(decode), median(prefill)
	return r
}

// progress aggregates pull bytes across layers. Layers are told apart only by
// their total size, so equal-sized layers share a slot; this only affects the
// displayed percentage.
type progress struct {
	out      io.Writer
	tag      string
	expected int64
	tty      bool
	layers   map[int64]int64 // total -> max completed
	lastPct  int
	started  bool
}

func newProgress(out io.Writer, tag string, expected uint64, tty bool) *progress {
	return &progress{out: out, tag: tag, expected: int64(expected), tty: tty, layers: map[int64]int64{}, lastPct: -1}
}

func (p *progress) update(completed, total int64) {
	if completed > p.layers[total] {
		p.layers[total] = completed
	} else if _, ok := p.layers[total]; !ok {
		p.layers[total] = completed
	}
	var done, sum int64
	for t, c := range p.layers {
		done += c
		sum += t
	}
	den := sum
	if p.expected > den {
		den = p.expected
	}
	pct := int(done * 100 / den)
	if pct > 99 {
		pct = 99 // 100 is reserved for the success line
	}
	p.show(pct)
}

func (p *progress) show(pct int) {
	if pct == p.lastPct {
		return
	}
	if p.tty {
		fmt.Fprintf(p.out, "\r%s %d%%", p.tag, pct)
		p.started = true
	} else if pct/10 > p.lastPct/10 || p.lastPct < 0 {
		fmt.Fprintf(p.out, "%s %d%%\n", p.tag, pct/10*10)
	}
	p.lastPct = pct
}

func (p *progress) finish(ok bool) {
	if ok {
		p.show(100)
	}
	if p.tty && p.started {
		fmt.Fprintln(p.out)
	}
}
