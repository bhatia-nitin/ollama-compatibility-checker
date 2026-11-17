package main

import (
	"reflect"
	"strings"
	"testing"
)

func TestIgnoredTagsAreSkipped(t *testing.T) {
	ignore := parseIgnore(strings.NewReader("# c\nllama4\nqwen3:4b-q8_0\n"))
	skip := skipTags(map[string]bool{"qwen3:8b-q4_K_M": true}, ignore)
	got := newQuantTags("qwen3", []string{"8b-q4_K_M", "4b-q8_0", "14b-q8_0"}, skip)
	if want := []string{"qwen3:14b-q8_0"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	if skip["llama4"] {
		t.Error("family names must not enter the tag skip set")
	}
}
