// Package bench talks to a local Ollama server for the opt-in bench command.
package bench

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

const defaultBaseURL = "http://127.0.0.1:11434"

// Client is a minimal Ollama HTTP API client.
type Client struct {
	BaseURL string
	// HTTP may be nil. The default has no overall timeout (pulls are long);
	// cancellation is controlled by the context.
	HTTP *http.Client
}

// HostFromEnv derives the API base URL from OLLAMA_HOST and reports whether
// the host is loopback. Accepts "host", "host:port" and "scheme://host:port".
func HostFromEnv() (baseURL string, loopback bool) {
	raw := strings.TrimSpace(os.Getenv("OLLAMA_HOST"))
	if raw == "" {
		return defaultBaseURL, true
	}
	scheme := "http"
	rest := raw
	if i := strings.Index(rest, "://"); i >= 0 {
		scheme, rest = rest[:i], rest[i+3:]
	}
	rest = strings.TrimSuffix(rest, "/")
	// Bare IPv6 without brackets (for example "::").
	if strings.Count(rest, ":") > 1 && !strings.HasPrefix(rest, "[") {
		rest = "[" + rest + "]"
	}
	u, err := url.Parse(scheme + "://" + rest)
	if err != nil || u.Hostname() == "" && !strings.HasPrefix(rest, ":") {
		return defaultBaseURL, true
	}
	host, port := u.Hostname(), u.Port()
	if port == "" {
		port = "11434"
	}
	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "127.0.0.1"
	}
	if ip := net.ParseIP(host); ip != nil {
		loopback = ip.IsLoopback()
	} else {
		loopback = strings.EqualFold(host, "localhost")
	}
	hostPort := net.JoinHostPort(host, port)
	return scheme + "://" + hostPort, loopback
}

func (c *Client) httpClient() *http.Client {
	if c.HTTP != nil {
		return c.HTTP
	}
	return http.DefaultClient
}

// do sends a request and returns the response; non-2xx becomes an error.
func (c *Client) do(ctx context.Context, method, path string, body any) (*http.Response, error) {
	var rd io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		rd = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, strings.TrimSuffix(c.BaseURL, "/")+path, rd)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.httpClient().Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		defer resp.Body.Close()
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		var e struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(data, &e) == nil && e.Error != "" {
			return nil, fmt.Errorf("ollama %s %s: status %d: %s", method, path, resp.StatusCode, e.Error)
		}
		return nil, fmt.Errorf("ollama %s %s: status %d", method, path, resp.StatusCode)
	}
	return resp, nil
}

func (c *Client) doJSON(ctx context.Context, method, path string, in, out any) error {
	resp, err := c.do(ctx, method, path, in)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if out == nil {
		_, _ = io.Copy(io.Discard, resp.Body)
		return nil
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return fmt.Errorf("ollama %s %s: decode response: %w", method, path, err)
	}
	return nil
}

// Version returns the Ollama server version.
func (c *Client) Version(ctx context.Context) (string, error) {
	var out struct {
		Version string `json:"version"`
	}
	if err := c.doJSON(ctx, http.MethodGet, "/api/version", nil, &out); err != nil {
		return "", err
	}
	return out.Version, nil
}

// Tags returns the names of locally installed models.
func (c *Client) Tags(ctx context.Context) ([]string, error) {
	var out struct {
		Models []struct {
			Name  string `json:"name"`
			Model string `json:"model"`
		} `json:"models"`
	}
	if err := c.doJSON(ctx, http.MethodGet, "/api/tags", nil, &out); err != nil {
		return nil, err
	}
	names := make([]string, 0, len(out.Models))
	for _, m := range out.Models {
		n := m.Name
		if n == "" {
			n = m.Model
		}
		if n != "" {
			names = append(names, n)
		}
	}
	return names, nil
}

// Pull downloads a model, calling progress(completed, total) for lines that
// report a total. It returns nil only after a "success" status line.
func (c *Client) Pull(ctx context.Context, tag string, progress func(completed, total int64)) error {
	resp, err := c.do(ctx, http.MethodPost, "/api/pull", map[string]any{"model": tag, "stream": true})
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	success := false
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var msg struct {
			Status    string `json:"status"`
			Total     int64  `json:"total"`
			Completed int64  `json:"completed"`
			Error     string `json:"error"`
		}
		if err := json.Unmarshal(line, &msg); err != nil {
			return fmt.Errorf("ollama pull: bad stream line: %w", err)
		}
		if msg.Error != "" {
			return fmt.Errorf("ollama pull %s: %s", tag, msg.Error)
		}
		if msg.Total > 0 && progress != nil {
			progress(msg.Completed, msg.Total)
		}
		if msg.Status == "success" {
			success = true
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := sc.Err(); err != nil {
		return err
	}
	if !success {
		return fmt.Errorf("ollama pull %s: stream ended without success", tag)
	}
	return nil
}

// GenResult holds timing counters from a non-streaming generate call.
type GenResult struct {
	EvalCount          int
	EvalDuration       time.Duration
	PromptEvalCount    int
	PromptEvalDuration time.Duration
}

// Generate runs one deterministic completion (temperature 0, seed 42).
func (c *Client) Generate(ctx context.Context, tag, prompt string, numCtx, numPredict int) (GenResult, error) {
	in := map[string]any{
		"model":  tag,
		"prompt": prompt,
		"stream": false,
		"options": map[string]any{
			"num_ctx":     numCtx,
			"num_predict": numPredict,
			"temperature": 0,
			"seed":        42,
		},
	}
	var out struct {
		EvalCount          int   `json:"eval_count"`
		EvalDuration       int64 `json:"eval_duration"`
		PromptEvalCount    int   `json:"prompt_eval_count"`
		PromptEvalDuration int64 `json:"prompt_eval_duration"`
	}
	if err := c.doJSON(ctx, http.MethodPost, "/api/generate", in, &out); err != nil {
		return GenResult{}, err
	}
	return GenResult{
		EvalCount:          out.EvalCount,
		EvalDuration:       time.Duration(out.EvalDuration),
		PromptEvalCount:    out.PromptEvalCount,
		PromptEvalDuration: time.Duration(out.PromptEvalDuration),
	}, nil
}

// PSModel is one entry of the running-models list.
type PSModel struct {
	Name     string
	Size     int64
	SizeVRAM int64
}

// PS lists models currently loaded in memory.
func (c *Client) PS(ctx context.Context) ([]PSModel, error) {
	var out struct {
		Models []struct {
			Name     string `json:"name"`
			Size     int64  `json:"size"`
			SizeVRAM int64  `json:"size_vram"`
		} `json:"models"`
	}
	if err := c.doJSON(ctx, http.MethodGet, "/api/ps", nil, &out); err != nil {
		return nil, err
	}
	res := make([]PSModel, 0, len(out.Models))
	for _, m := range out.Models {
		res = append(res, PSModel{Name: m.Name, Size: m.Size, SizeVRAM: m.SizeVRAM})
	}
	return res, nil
}

// Delete removes a local model.
func (c *Client) Delete(ctx context.Context, tag string) error {
	return c.doJSON(ctx, http.MethodDelete, "/api/delete", map[string]string{"model": tag}, nil)
}
