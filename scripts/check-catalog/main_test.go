package main

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func fixture(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestParseLibrary(t *testing.T) {
	got, err := parseLibrary(fixture(t, "library.html"))
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"bge-m3", "deepseek-r1", "gemma4", "llama3.1", "nomic-embed-text", "qwen3"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v (tag links like /library/qwen3:0.6b must not count)", got, want)
	}
}

func TestParseLibraryEmpty(t *testing.T) {
	_, err := parseLibrary(fixture(t, "empty.html"))
	if err == nil || !strings.Contains(err.Error(), "page format changed") {
		t.Fatalf("want page-format error, got %v", err)
	}
}

func TestParseTags(t *testing.T) {
	got, err := parseTags("qwen3", fixture(t, "qwen3-tags.html"))
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"0.6b", "0.6b-fp16", "0.6b-q4_K_M", "0.6b-q8_0", "1.7b-q4_K_M", "8b-q8_0", "latest"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v (qwen3-coder: must not leak in)", got, want)
	}
}

func TestParseTagsEmpty(t *testing.T) {
	_, err := parseTags("qwen3", fixture(t, "empty.html"))
	if err == nil || !strings.Contains(err.Error(), "page format changed") {
		t.Fatalf("want page-format error, got %v", err)
	}
}

func TestQuantPolicy(t *testing.T) {
	for tag, want := range map[string]bool{
		"8b-q4_K_M":          true,
		"70b-q8_0":           true,
		"32b-q3_K_M":         true,
		"8b-instruct-q4_K_M": true,
		"8b-fp16":            false,
		"8b":                 false,
		"latest":             false,
		"q4_K_M":             false,
		"8b-q4_K_S":          false,
		"8b-q5_K_M":          false,
	} {
		if got := quantPolicy(tag); got != want {
			t.Errorf("quantPolicy(%q) = %v, want %v", tag, got, want)
		}
	}
}

func TestNewQuantTags(t *testing.T) {
	pageTags, err := parseTags("qwen3", fixture(t, "qwen3-tags.html"))
	if err != nil {
		t.Fatal(err)
	}
	have := map[string]bool{"qwen3:0.6b-q4_K_M": true, "qwen3:0.6b-q8_0": true}
	got := newQuantTags("qwen3", pageTags, have)
	want := []string{"qwen3:1.7b-q4_K_M", "qwen3:8b-q8_0"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestNewFamiliesAndIgnore(t *testing.T) {
	lib, err := parseLibrary(fixture(t, "library.html"))
	if err != nil {
		t.Fatal(err)
	}
	ignore := parseIgnore(strings.NewReader("# embeddings\nbge-m3  # trailing\n\nnomic-embed-text\n"))
	tracked := map[string]bool{"qwen3": true, "llama3.1": true, "deepseek-r1": true}
	got := newFamilies(lib, tracked, ignore)
	if want := []string{"gemma4"}; !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestMarkdown(t *testing.T) {
	var r report
	r.NewTags = map[string][]string{}
	if md := r.markdown(); !strings.Contains(md, "No drift found.") {
		t.Errorf("clean report: %s", md)
	}
	var many []string
	for i := 0; i < perFamilyCap+5; i++ {
		many = append(many, "qwen3:x")
	}
	r = report{Missing: []string{"a:b"}, NewTags: map[string][]string{"qwen3": many}, NewFamilies: []string{"gemma4"}}
	md := r.markdown()
	for _, s := range []string{"`a:b`", "... and 5 more", "[gemma4](https://ollama.com/library/gemma4)"} {
		if !strings.Contains(md, s) {
			t.Errorf("report missing %q:\n%s", s, md)
		}
	}
}
