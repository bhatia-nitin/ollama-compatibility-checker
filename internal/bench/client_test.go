package bench

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestHostFromEnv(t *testing.T) {
	tests := []struct {
		name     string
		env      string
		set      bool
		wantURL  string
		wantLoop bool
	}{
		{"unset", "", false, "http://127.0.0.1:11434", true},
		{"bind all v4", "0.0.0.0", true, "http://127.0.0.1:11434", true},
		{"bind all v4 port", "0.0.0.0:11434", true, "http://127.0.0.1:11434", true},
		{"bind all v6", "::", true, "http://127.0.0.1:11434", true},
		{"remote host port", "gpu-box:11434", true, "http://gpu-box:11434", false},
		{"remote host only", "gpu-box", true, "http://gpu-box:11434", false},
		{"localhost", "localhost:9999", true, "http://localhost:9999", true},
		{"full url", "http://h:1234", true, "http://h:1234", false},
		{"https default port", "https://example.com", true, "https://example.com:11434", false},
		{"v6 loopback", "http://[::1]:11434", true, "http://[::1]:11434", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.set {
				t.Setenv("OLLAMA_HOST", tt.env)
			} else {
				t.Setenv("OLLAMA_HOST", "")
			}
			got, loop := HostFromEnv()
			if got != tt.wantURL || loop != tt.wantLoop {
				t.Errorf("got (%q, %v), want (%q, %v)", got, loop, tt.wantURL, tt.wantLoop)
			}
		})
	}
}

func TestVersionAndTags(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/version":
			fmt.Fprint(w, `{"version":"0.5.1"}`)
		case "/api/tags":
			fmt.Fprint(w, `{"models":[{"name":"llama3.2:3b"},{"model":"qwen2.5:7b"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	c := &Client{BaseURL: srv.URL}
	v, err := c.Version(context.Background())
	if err != nil || v != "0.5.1" {
		t.Fatalf("version = %q, %v", v, err)
	}
	tags, err := c.Tags(context.Background())
	if err != nil || len(tags) != 2 || tags[0] != "llama3.2:3b" || tags[1] != "qwen2.5:7b" {
		t.Fatalf("tags = %v, %v", tags, err)
	}
}

func TestNon2xxError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		fmt.Fprint(w, `{"error":"model not found"}`)
	}))
	defer srv.Close()
	c := &Client{BaseURL: srv.URL}
	_, err := c.Generate(context.Background(), "x", "hi", 2048, 8)
	if err == nil || !strings.Contains(err.Error(), "404") || !strings.Contains(err.Error(), "model not found") {
		t.Fatalf("err = %v", err)
	}
}

func TestPullStreamsProgress(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model  string `json:"model"`
			Stream bool   `json:"stream"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		if r.Method != http.MethodPost || r.URL.Path != "/api/pull" || body.Model != "m:1" || !body.Stream {
			t.Errorf("bad request: %s %s %+v", r.Method, r.URL.Path, body)
		}
		fmt.Fprintln(w, `{"status":"pulling manifest"}`)
		fmt.Fprintln(w, `{"status":"downloading","digest":"d","total":100,"completed":10}`)
		fmt.Fprintln(w, `{"status":"downloading","digest":"d","total":100,"completed":50}`)
		fmt.Fprintln(w, `{"status":"downloading","digest":"d","total":100,"completed":100}`)
		fmt.Fprintln(w, `{"status":"success"}`)
	}))
	defer srv.Close()
	c := &Client{BaseURL: srv.URL}
	var got []int64
	err := c.Pull(context.Background(), "m:1", func(done, total int64) {
		if total != 100 {
			t.Errorf("total = %d", total)
		}
		got = append(got, done)
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0] != 10 || got[1] != 50 || got[2] != 100 {
		t.Errorf("progress = %v", got)
	}
}

func TestPullErrorLine(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"status":"pulling manifest"}`)
		fmt.Fprintln(w, `{"error":"pull model manifest: file does not exist"}`)
	}))
	defer srv.Close()
	c := &Client{BaseURL: srv.URL}
	err := c.Pull(context.Background(), "nope", nil)
	if err == nil || !strings.Contains(err.Error(), "pull model manifest: file does not exist") {
		t.Fatalf("err = %v", err)
	}
}

func TestPullNoSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"status":"pulling manifest"}`)
	}))
	defer srv.Close()
	c := &Client{BaseURL: srv.URL}
	if err := c.Pull(context.Background(), "m", nil); err == nil {
		t.Fatal("want error when stream ends without success")
	}
}

func TestPullCancel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, `{"status":"downloading","total":100,"completed":1}`)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer srv.Close()
	c := &Client{BaseURL: srv.URL}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- c.Pull(ctx, "m", func(_, _ int64) { cancel() })
	}()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Pull did not return promptly after cancel")
	}
}

func TestGenerateParsesDurations(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		opts, _ := body["options"].(map[string]any)
		if body["model"] != "m" || body["prompt"] != "hi" || body["stream"] != false ||
			opts["num_ctx"] != float64(4096) || opts["num_predict"] != float64(64) ||
			opts["temperature"] != float64(0) || opts["seed"] != float64(42) {
			t.Errorf("bad body: %v", body)
		}
		fmt.Fprint(w, `{"eval_count":64,"eval_duration":2000000000,"prompt_eval_count":9,"prompt_eval_duration":150000000}`)
	}))
	defer srv.Close()
	c := &Client{BaseURL: srv.URL}
	res, err := c.Generate(context.Background(), "m", "hi", 4096, 64)
	if err != nil {
		t.Fatal(err)
	}
	if res.EvalCount != 64 || res.EvalDuration != 2*time.Second ||
		res.PromptEvalCount != 9 || res.PromptEvalDuration != 150*time.Millisecond {
		t.Errorf("res = %+v", res)
	}
}

func TestPS(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/ps" {
			t.Errorf("path %s", r.URL.Path)
		}
		fmt.Fprint(w, `{"models":[{"name":"m:1","size":5000,"size_vram":4000}]}`)
	}))
	defer srv.Close()
	c := &Client{BaseURL: srv.URL}
	ps, err := c.PS(context.Background())
	if err != nil || len(ps) != 1 || ps[0] != (PSModel{Name: "m:1", Size: 5000, SizeVRAM: 4000}) {
		t.Fatalf("ps = %v, %v", ps, err)
	}
}

func TestDeleteSendsBody(t *testing.T) {
	var gotMethod, gotPath, gotBody string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		gotMethod, gotPath, gotBody = r.Method, r.URL.Path, strings.TrimSpace(string(b))
	}))
	defer srv.Close()
	c := &Client{BaseURL: srv.URL}
	if err := c.Delete(context.Background(), "m:1"); err != nil {
		t.Fatal(err)
	}
	if gotMethod != http.MethodDelete || gotPath != "/api/delete" || gotBody != `{"model":"m:1"}` {
		t.Errorf("got %s %s %s", gotMethod, gotPath, gotBody)
	}
}
