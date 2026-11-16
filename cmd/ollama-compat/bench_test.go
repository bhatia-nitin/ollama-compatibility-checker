package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/shra26/ollama-compatibility-checker/internal/bench"
)

const (
	tagSmall = "llama3.2:3b-instruct-q4_K_M"
	tagBig   = "qwen2.5:14b-instruct-q4_K_M"
)

type benchFake struct {
	mu   sync.Mutex
	reqs []string
}

func (b *benchFake) has(r string) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	for _, x := range b.reqs {
		if x == r {
			return true
		}
	}
	return false
}

func newBenchFake(t *testing.T, installed ...string) (*benchFake, *httptest.Server) {
	t.Helper()
	bf := &benchFake{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bf.mu.Lock()
		bf.reqs = append(bf.reqs, r.Method+" "+r.URL.Path)
		bf.mu.Unlock()
		switch r.URL.Path {
		case "/api/version":
			fmt.Fprint(w, `{"version":"0.5.1"}`)
		case "/api/tags":
			var ms []string
			for _, n := range installed {
				ms = append(ms, fmt.Sprintf(`{"name":%q}`, n))
			}
			fmt.Fprintf(w, `{"models":[%s]}`, strings.Join(ms, ","))
		case "/api/generate":
			fmt.Fprint(w, `{"eval_count":400,"eval_duration":10000000000,"prompt_eval_count":60,"prompt_eval_duration":1000000000}`)
		case "/api/ps":
			fmt.Fprintf(w, `{"models":[{"name":%q,"size":100,"size_vram":100}]}`, tagSmall)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return bf, srv
}

func runBench(t *testing.T, srv *httptest.Server, tty bool, stdin string, args ...string) (string, string, error) {
	t.Helper()
	var out, errb bytes.Buffer
	env := benchEnv{
		client: func() (*bench.Client, bool) { return &bench.Client{BaseURL: srv.URL}, true },
		isTTY:  func() bool { return tty },
	}
	cmd := newRootCmdEnv(&out, &errb, fakeDetect, env)
	cmd.SetIn(strings.NewReader(stdin))
	cmd.SetArgs(append([]string{"bench", "--catalog", cat}, args...))
	err := cmd.Execute()
	return out.String(), errb.String(), err
}

func TestBenchInstalledOnlyText(t *testing.T) {
	bf, srv := newBenchFake(t, tagSmall)
	out, errs, err := runBench(t, srv, false, "", "--installed-only")
	if err != nil {
		t.Fatal(err)
	}
	if bf.has("POST /api/pull") {
		t.Error("pulled")
	}
	for _, want := range []string{"est. tok/s", "measured tok/s", "prefill tok/s", "GPU est/measured", tagSmall, "40.0", "Measured ranking", "skipped (not installed)"} {
		if !strings.Contains(out, want) {
			t.Errorf("stdout missing %q:\n%s", want, out)
		}
	}
	if !strings.Contains(errs, tagBig+": skipped (not installed)") {
		t.Errorf("stderr = %q", errs)
	}
}

func TestBenchJSON(t *testing.T) {
	_, srv := newBenchFake(t, tagSmall)
	out, _, err := runBench(t, srv, false, "", "--installed-only", "--json")
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		SchemaVersion int            `json:"schema_version"`
		Hardware      map[string]any `json:"hardware"`
		Results       []struct {
			Tag          string  `json:"tag"`
			MeasuredTokS float64 `json:"measured_tok_s"`
			PrefillTokS  float64 `json:"prefill_tok_s"`
			Error        string  `json:"error"`
		} `json:"results"`
	}
	if err := json.Unmarshal([]byte(out), &doc); err != nil {
		t.Fatalf("invalid json: %v\n%s", err, out)
	}
	if doc.SchemaVersion != 1 || doc.Hardware["os"] != "linux" || len(doc.Results) != 2 {
		t.Fatalf("doc = %+v", doc)
	}
	if doc.Results[0].Tag != tagSmall || doc.Results[0].MeasuredTokS != 40 || doc.Results[0].PrefillTokS != 60 {
		t.Errorf("result 0 = %+v", doc.Results[0])
	}
	if doc.Results[1].Error != "skipped (not installed)" {
		t.Errorf("result 1 = %+v", doc.Results[1])
	}
}

func TestBenchRefusesNonTTYWithoutYes(t *testing.T) {
	bf, srv := newBenchFake(t)
	_, _, err := runBench(t, srv, false, "y\n")
	if err == nil || !strings.Contains(err.Error(), "refusing to download without --yes") {
		t.Fatalf("err = %v", err)
	}
	if bf.has("POST /api/pull") {
		t.Error("pulled")
	}
}

func TestBenchFlagErrors(t *testing.T) {
	_, srv := newBenchFake(t)
	for _, args := range [][]string{{"--max-download", "abc"}, {"--runs", "0"}, {"--ctx", "0"}} {
		if _, _, err := runBench(t, srv, false, "", args...); err == nil {
			t.Errorf("%v: want error", args)
		}
	}
}

func TestBenchMaxDownloadCap(t *testing.T) {
	bf, srv := newBenchFake(t)
	_, _, err := runBench(t, srv, true, "y\n", "--max-download", "1GB")
	if err == nil || !strings.Contains(err.Error(), "--max-download") {
		t.Fatalf("err = %v", err)
	}
	if bf.has("POST /api/pull") {
		t.Error("pulled")
	}
}
