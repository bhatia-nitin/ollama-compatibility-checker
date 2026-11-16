package bench

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/shra26/ollama-compatibility-checker/internal/catalog"
	"github.com/shra26/ollama-compatibility-checker/internal/estimate"
)

const gbyte = 1_000_000_000

func est(tag string, sizeBytes uint64, gpuFrac, tokS float64) estimate.Estimate {
	return estimate.Estimate{
		Model:       catalog.Model{Name: tag},
		Variant:     catalog.Variant{Tag: tag, SizeBytes: sizeBytes},
		GPUFraction: gpuFrac, TokS: tokS,
	}
}

// fake is a recording fake Ollama server.
type fake struct {
	mu        sync.Mutex
	reqs      []string
	installed []string
	rates     []float64 // decode tok/s handed out to non-warm-up generates
	genCalls  []string  // model of every generate call
	psSize    int64
	psVRAM    int64
	genErr    map[string]bool // models whose generate fails
	onPull    func(w http.ResponseWriter, r *http.Request)
	onGen     func(r *http.Request)
	deleted   []string
	srv       *httptest.Server
}

func newFake(t *testing.T, installed ...string) *fake {
	t.Helper()
	f := &fake{installed: installed, psSize: 100, psVRAM: 100, genErr: map[string]bool{}}
	f.srv = httptest.NewServer(http.HandlerFunc(f.handle))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fake) client() *Client { return &Client{BaseURL: f.srv.URL} }

func (f *fake) handle(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	f.reqs = append(f.reqs, r.Method+" "+r.URL.Path)
	f.mu.Unlock()
	switch r.URL.Path {
	case "/api/version":
		fmt.Fprint(w, `{"version":"0.5.1"}`)
	case "/api/tags":
		type m struct {
			Name string `json:"name"`
		}
		var ms []m
		f.mu.Lock()
		for _, n := range f.installed {
			ms = append(ms, m{n})
		}
		f.mu.Unlock()
		_ = json.NewEncoder(w).Encode(map[string]any{"models": ms})
	case "/api/pull":
		if f.onPull != nil {
			f.onPull(w, r)
			return
		}
		var in struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(r.Body).Decode(&in)
		fmt.Fprintln(w, `{"status":"pulling abc","total":100,"completed":50}`)
		fmt.Fprintln(w, `{"status":"pulling abc","total":100,"completed":100}`)
		fmt.Fprintln(w, `{"status":"success"}`)
		f.mu.Lock()
		f.installed = append(f.installed, in.Model)
		f.mu.Unlock()
	case "/api/generate":
		var in struct {
			Model   string `json:"model"`
			Options struct {
				NumPredict int `json:"num_predict"`
			} `json:"options"`
		}
		_ = json.NewDecoder(r.Body).Decode(&in)
		f.mu.Lock()
		f.genCalls = append(f.genCalls, in.Model)
		fail := f.genErr[in.Model]
		rate := 10.0
		if in.Options.NumPredict != 16 && len(f.rates) > 0 {
			rate, f.rates = f.rates[0], f.rates[1:]
		}
		f.mu.Unlock()
		if f.onGen != nil {
			f.onGen(r)
		}
		if fail {
			http.Error(w, `{"error":"boom"}`, 500)
			return
		}
		// eval_count = rate * 10 over 10 s => decode rate tok/s.
		fmt.Fprintf(w, `{"eval_count":%d,"eval_duration":10000000000,"prompt_eval_count":60,"prompt_eval_duration":1000000000}`, int(rate*10))
	case "/api/ps":
		f.mu.Lock()
		last := ""
		if len(f.genCalls) > 0 {
			last = f.genCalls[len(f.genCalls)-1]
		}
		f.mu.Unlock()
		fmt.Fprintf(w, `{"models":[{"name":%q,"size":%d,"size_vram":%d}]}`, last, f.psSize, f.psVRAM)
	case "/api/delete":
		var in struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(r.Body).Decode(&in)
		f.mu.Lock()
		f.deleted = append(f.deleted, in.Model)
		f.mu.Unlock()
	default:
		http.NotFound(w, r)
	}
}

func (f *fake) count(req string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	n := 0
	for _, r := range f.reqs {
		if r == req {
			n++
		}
	}
	return n
}

func opts(in string, out *bytes.Buffer) Options {
	return Options{In: strings.NewReader(in), Out: out, IsTTY: true, Loopback: true, Ctx: 4096, Runs: 3}
}

func TestRefusesNonTTYWithoutYes(t *testing.T) {
	f := newFake(t)
	var out bytes.Buffer
	o := opts("y\n", &out)
	o.IsTTY = false
	_, err := Run(context.Background(), f.client(), []estimate.Estimate{est("m:1b", gbyte, 1, 50)}, o)
	if err == nil || !strings.Contains(err.Error(), "refusing to download without --yes") {
		t.Fatalf("err = %v", err)
	}
	if got := strings.Join(f.reqs, ","); got != "GET /api/version,GET /api/tags" {
		t.Errorf("requests = %s", got)
	}
	if strings.Contains(out.String(), "[y/N]") {
		t.Errorf("prompted on non-TTY: %q", out.String())
	}
}

func TestPromptDefaultNo(t *testing.T) {
	for _, in := range []string{"\n", "", "n\n", "maybe\n"} {
		f := newFake(t)
		var out bytes.Buffer
		res, err := Run(context.Background(), f.client(), []estimate.Estimate{est("m:1b", gbyte, 1, 50)}, opts(in, &out))
		if err != nil || res != nil {
			t.Fatalf("in %q: res=%v err=%v", in, res, err)
		}
		if f.count("POST /api/pull") != 0 {
			t.Errorf("in %q: pulled", in)
		}
		if !strings.Contains(out.String(), "Cancelled.") {
			t.Errorf("in %q: no Cancelled: %q", in, out.String())
		}
	}
}

func TestPromptShowsTotal(t *testing.T) {
	f := newFake(t, "have:7b")
	var out bytes.Buffer
	sl := []estimate.Estimate{est("have:7b", 9*gbyte, 1, 30), est("need:3b", 2_500_000_000, 1, 60), est("need:8b", 4_000_000_000, 1, 40)}
	_, err := Run(context.Background(), f.client(), sl, opts("n\n", &out))
	if err != nil {
		t.Fatal(err)
	}
	s := out.String()
	if !strings.Contains(s, "Download 6.5 GB and benchmark 3 models? [y/N] ") {
		t.Errorf("prompt wrong: %q", s)
	}
	if strings.Contains(s, "15.5") {
		t.Errorf("installed model counted in total: %q", s)
	}
}

func TestMaxDownloadCap(t *testing.T) {
	f := newFake(t)
	var out bytes.Buffer
	o := opts("y\n", &out)
	o.MaxDownload = 20 * gbyte
	_, err := Run(context.Background(), f.client(), []estimate.Estimate{est("big:70b", 21*gbyte, 1, 5)}, o)
	if err == nil || !strings.Contains(err.Error(), "21.0 GB") || !strings.Contains(err.Error(), "20.0 GB") {
		t.Fatalf("err = %v", err)
	}
	if f.count("POST /api/pull") != 0 {
		t.Error("pulled")
	}
}

func TestInstalledOnly(t *testing.T) {
	f := newFake(t, "have:7b")
	f.rates = []float64{30, 30, 30}
	var out bytes.Buffer
	o := opts("", &out)
	o.InstalledOnly = true
	res, err := Run(context.Background(), f.client(), []estimate.Estimate{est("have:7b", gbyte, 1, 30), est("miss:3b", gbyte, 1, 60)}, o)
	if err != nil {
		t.Fatal(err)
	}
	if f.count("POST /api/pull") != 0 {
		t.Error("pulled")
	}
	if !strings.Contains(out.String(), "miss:3b: skipped (not installed)") {
		t.Errorf("missing skip line: %q", out.String())
	}
	if len(res) != 2 || res[0].DecodeTokS != 30 || res[1].Err != "skipped (not installed)" {
		t.Errorf("res = %+v", res)
	}
	if strings.Contains(out.String(), "[y/N]") {
		t.Error("prompted")
	}
}

func TestLatestEquivalence(t *testing.T) {
	f := newFake(t, "llama3:latest")
	var out bytes.Buffer
	o := opts("", &out)
	o.InstalledOnly = true
	f.rates = []float64{10, 10, 10}
	res, err := Run(context.Background(), f.client(), []estimate.Estimate{est("llama3", gbyte, 1, 30)}, o)
	if err != nil || len(res) != 1 || res[0].Err != "" {
		t.Fatalf("res=%+v err=%v", res, err)
	}
}

func TestRemoteRefused(t *testing.T) {
	f := newFake(t, "have:7b")
	var out bytes.Buffer
	o := opts("", &out)
	o.Loopback = false
	o.InstalledOnly = true
	_, err := Run(context.Background(), f.client(), []estimate.Estimate{est("have:7b", gbyte, 1, 30)}, o)
	if err == nil || !strings.Contains(err.Error(), "--allow-remote") {
		t.Fatalf("err = %v", err)
	}
	if len(f.reqs) != 0 {
		t.Errorf("requests before refusal: %v", f.reqs)
	}
	o.AllowRemote = true
	if _, err := Run(context.Background(), f.client(), []estimate.Estimate{est("have:7b", gbyte, 1, 30)}, o); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "describe the remote machine") {
		t.Errorf("no warning: %q", out.String())
	}
}

func TestUnreachableHint(t *testing.T) {
	f := newFake(t)
	c := f.client()
	f.srv.Close()
	var out bytes.Buffer
	_, err := Run(context.Background(), c, []estimate.Estimate{est("m:1b", gbyte, 1, 50)}, opts("y\n", &out))
	if err == nil || !strings.Contains(err.Error(), "https://ollama.com/download") {
		t.Fatalf("err = %v", err)
	}
}

func TestMedianOfRuns(t *testing.T) {
	f := newFake(t, "m:1b")
	f.rates = []float64{30, 50, 40}
	var out bytes.Buffer
	o := opts("", &out)
	o.InstalledOnly = true
	res, err := Run(context.Background(), f.client(), []estimate.Estimate{est("m:1b", gbyte, 1, 40)}, o)
	if err != nil {
		t.Fatal(err)
	}
	if res[0].DecodeTokS != 40 {
		t.Errorf("decode = %v", res[0].DecodeTokS)
	}
	if res[0].PrefillTokS != 60 {
		t.Errorf("prefill = %v", res[0].PrefillTokS)
	}
	if n := len(f.genCalls); n != 4 {
		t.Errorf("generate calls = %d, want 1 warm-up + 3 runs", n)
	}
}

func TestMedianEven(t *testing.T) {
	if got := median([]float64{10, 40, 20, 30}); got != 25 {
		t.Errorf("median = %v", got)
	}
}

func TestMismatchFlag(t *testing.T) {
	f := newFake(t, "m:1b")
	f.psSize, f.psVRAM = 100, 60
	var out bytes.Buffer
	o := opts("", &out)
	o.InstalledOnly = true
	res, err := Run(context.Background(), f.client(), []estimate.Estimate{est("m:1b", gbyte, 1.0, 40)}, o)
	if err != nil {
		t.Fatal(err)
	}
	if !res[0].Mismatch || res[0].MeasuredGPUFraction != 0.6 {
		t.Errorf("res = %+v", res[0])
	}
}

func TestGenerateErrorContinues(t *testing.T) {
	f := newFake(t, "bad:1b", "ok:1b")
	f.genErr["bad:1b"] = true
	var out bytes.Buffer
	o := opts("", &out)
	o.InstalledOnly = true
	res, err := Run(context.Background(), f.client(), []estimate.Estimate{est("bad:1b", gbyte, 1, 40), est("ok:1b", gbyte, 1, 40)}, o)
	if err != nil {
		t.Fatal(err)
	}
	if res[0].Err == "" || res[1].Err != "" || res[1].DecodeTokS == 0 {
		t.Errorf("res = %+v", res)
	}
}

func TestYesPullsWithoutPrompt(t *testing.T) {
	f := newFake(t)
	var out bytes.Buffer
	o := opts("", &out)
	o.IsTTY, o.Yes = false, true
	res, err := Run(context.Background(), f.client(), []estimate.Estimate{est("m:1b", gbyte, 1, 40)}, o)
	if err != nil {
		t.Fatal(err)
	}
	if f.count("POST /api/pull") != 1 || !res[0].Pulled {
		t.Errorf("pulls=%d res=%+v", f.count("POST /api/pull"), res)
	}
	if strings.Contains(out.String(), "[y/N]") {
		t.Error("prompted with --yes")
	}
	// Non-TTY or --yes never deletes on its own; it prints the commands.
	if len(f.deleted) != 0 || !strings.Contains(out.String(), "ollama rm m:1b") {
		t.Errorf("deleted=%v out=%q", f.deleted, out.String())
	}
}

func TestCleanupOnlyPulled(t *testing.T) {
	sl := []estimate.Estimate{est("have:7b", gbyte, 1, 30), est("new:3b", 2*gbyte, 1, 60)}
	f := newFake(t, "have:7b")
	var out bytes.Buffer
	// "y" confirms the download, the empty line takes the default Yes at cleanup.
	_, err := Run(context.Background(), f.client(), sl, opts("y\n\n", &out))
	if err != nil {
		t.Fatal(err)
	}
	if len(f.deleted) != 1 || f.deleted[0] != "new:3b" {
		t.Errorf("deleted = %v", f.deleted)
	}
	if !strings.Contains(out.String(), "Remove the 1 models downloaded for this benchmark to free 2.0 GB? [Y/n] ") {
		t.Errorf("cleanup prompt: %q", out.String())
	}

	f2 := newFake(t, "have:7b")
	var out2 bytes.Buffer
	o := opts("y\n", &out2)
	o.Keep = true
	if _, err := Run(context.Background(), f2.client(), sl, o); err != nil {
		t.Fatal(err)
	}
	if len(f2.deleted) != 0 || strings.Contains(out2.String(), "[Y/n]") {
		t.Errorf("keep: deleted=%v out=%q", f2.deleted, out2.String())
	}

	f3 := newFake(t, "have:7b")
	var out3 bytes.Buffer
	if _, err := Run(context.Background(), f3.client(), sl, opts("y\nn\n", &out3)); err != nil {
		t.Fatal(err)
	}
	if len(f3.deleted) != 0 {
		t.Errorf("n deleted %v", f3.deleted)
	}
}

func TestCancelDuringPull(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	f := newFake(t)
	f.onPull = func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"status":"pulling abc","total":100,"completed":10}`)
		w.(http.Flusher).Flush()
		cancel()
		<-r.Context().Done()
	}
	var out bytes.Buffer
	_, err := Run(ctx, f.client(), []estimate.Estimate{est("m:1b", gbyte, 1, 40)}, opts("y\n\n", &out))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	if f.count("POST /api/generate") != 0 || f.count("DELETE /api/delete") != 0 {
		t.Errorf("work after cancel: %v", f.reqs)
	}
	if strings.Contains(out.String(), "[Y/n]") {
		t.Errorf("cleanup prompt after cancel: %q", out.String())
	}
	if strings.Contains(out.String(), "ollama rm m:1b") {
		t.Errorf("incomplete pull listed for removal: %q", out.String())
	}
}

func TestCancelDuringBenchListsPulled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	f := newFake(t)
	f.onGen = func(r *http.Request) {
		cancel() // cancel while the warm-up generate is in flight
		<-r.Context().Done()
	}
	var out bytes.Buffer
	_, err := Run(ctx, f.client(), []estimate.Estimate{est("m:1b", gbyte, 1, 40)}, opts("y\n\n", &out))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	if !strings.Contains(out.String(), "Pulled during this run:") || !strings.Contains(out.String(), "ollama rm m:1b") {
		t.Errorf("out = %q", out.String())
	}
	if strings.Contains(out.String(), "[Y/n]") || len(f.deleted) != 0 {
		t.Errorf("cleanup after cancel: %q %v", out.String(), f.deleted)
	}
}

func TestParseSize(t *testing.T) {
	ok := map[string]uint64{
		"20GB": 20e9, "20gb": 20e9, " 20 GB ": 20e9, "1.5GiB": 1610612736, "512MB": 512e6, "1TB": 1e12,
		"2KiB": 2048, "3MiB": 3 << 20, "1TiB": 1 << 40, "7KB": 7000, "100": 100, "5B": 5, "0.5GB": 5e8,
	}
	for in, want := range ok {
		got, err := ParseSize(in)
		if err != nil || got != want {
			t.Errorf("ParseSize(%q) = %d, %v; want %d", in, got, err, want)
		}
	}
	for _, in := range []string{"abc", "", "-5GB", "GB", "5XB", "1.2.3GB", "nan", "inf", "99999999999999999999TB"} {
		if _, err := ParseSize(in); err == nil {
			t.Errorf("ParseSize(%q): want error", in)
		}
	}
}

func TestBenchPromptNoEmDash(t *testing.T) {
	if strings.ContainsRune(BenchPrompt, '—') || len(strings.Fields(BenchPrompt)) < 30 {
		t.Errorf("bad prompt: %q", BenchPrompt)
	}
}
