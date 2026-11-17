// Command gen-catalog builds internal/catalog/models.json from
// catalog/sources.yaml and the public Ollama registry.
//
// It is a dev-only tool and is not part of the shipped binary. For each tag it
// fetches the registry manifest (model layer size) and the first bytes of the
// GGUF blob (architecture metadata). Requests are sequential.
//
//	go run ./scripts/gen-catalog -in catalog/sources.yaml -out internal/catalog/models.json
package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/shra26/ollama-compatibility-checker/internal/catalog"
)

const (
	registry       = "https://registry.ollama.ai/v2/library/"
	modelMediaType = "application/vnd.ollama.image.model"
	firstRange     = 8 << 20  // 8 MiB covers most GGUF headers
	maxRange       = 64 << 20 // retry size when tokenizer arrays are larger
)

// source is one entry of catalog/sources.yaml.
type source struct {
	Family        string   `yaml:"family"`
	Name          string   `yaml:"name"`
	ParamsB       float64  `yaml:"params_b"`
	ActiveParamsB *float64 `yaml:"active_params_b"`
	Quality       int      `yaml:"quality"`
	QualitySource string   `yaml:"quality_source"`
	KVQuantOK     *bool    `yaml:"kv_quant_ok"`
	Tasks         []string `yaml:"tasks"`
	Quant         string   `yaml:"quant"` // for tags with no quant suffix whose GGUF file type has no name
	Tags          []string `yaml:"tags"`
}

type manifest struct {
	Layers []struct {
		MediaType string `json:"mediaType"`
		Digest    string `json:"digest"`
		Size      uint64 `json:"size"`
	} `json:"layers"`
}

var client = &http.Client{Timeout: 2 * time.Minute}

func main() {
	in := flag.String("in", "catalog/sources.yaml", "input source list")
	out := flag.String("out", "internal/catalog/models.json", "output catalog JSON")
	flag.Parse()
	log.SetFlags(0)

	raw, err := os.ReadFile(*in)
	if err != nil {
		log.Fatal(err)
	}
	var srcs []source
	if err := yaml.Unmarshal(raw, &srcs); err != nil {
		log.Fatalf("%s: %v", *in, err)
	}

	var models []catalog.Model
	for _, s := range srcs {
		m, err := build(s)
		if err != nil {
			log.Printf("DROP model %s: %v", s.Name, err)
			continue
		}
		models = append(models, m)
	}
	sort.SliceStable(models, func(i, j int) bool {
		a, b := models[i], models[j]
		if a.Family != b.Family {
			return a.Family < b.Family
		}
		if a.ParamsB != b.ParamsB {
			return a.ParamsB < b.ParamsB
		}
		return a.Name < b.Name
	})
	if err := catalog.Validate(models); err != nil {
		log.Fatal(err)
	}
	js, err := json.MarshalIndent(models, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(*out, append(js, '\n'), 0o644); err != nil {
		log.Fatal(err)
	}
	log.Printf("wrote %d models to %s", len(models), *out)
}

// build resolves every tag of one source entry. Tags that 404 are dropped.
func build(s source) (catalog.Model, error) {
	m := catalog.Model{
		Family:    s.Family,
		Name:      s.Name,
		ParamsB:   s.ParamsB,
		Quality:   s.Quality,
		KVQuantOK: s.KVQuantOK == nil || *s.KVQuantOK,
		Tasks:     s.Tasks,
	}
	if s.ParamsB <= 0 {
		return m, errors.New("params_b must be > 0")
	}
	var info *archInfo
	for _, tag := range s.Tags {
		layer, err := fetchManifest(tag)
		if errors.Is(err, errNotFound) {
			log.Printf("DROP tag %s: 404", tag)
			continue
		}
		if err != nil {
			return m, fmt.Errorf("%s: %w", tag, err)
		}
		quant := quantFromTag(tag)
		if info == nil || quant == "" {
			ai, err := fetchArchInfo(tag, layer.Digest)
			if err != nil {
				return m, fmt.Errorf("%s: %w", tag, err)
			}
			if info == nil {
				info = &ai
			}
			if quant == "" && s.Quant != "" {
				quant = s.Quant
			}
			if quant == "" {
				quant = quantName(tag, ai.FileType)
			}
		}
		m.Variants = append(m.Variants, catalog.Variant{
			Tag:        tag,
			SizeBytes:  layer.Size,
			Bits:       bits(layer.Size, s.ParamsB),
			Quant:      quant,
			ExtraBytes: layer.Extra,
		})
		log.Printf("ok %s %d bytes %s", tag, layer.Size, quant)
	}
	if info == nil {
		return m, errors.New("no tag resolved")
	}
	m.Layers, m.KVHeads, m.HeadDim, m.MaxCtx = info.Layers, info.KVHeads, info.HeadDim, info.MaxCtx
	m.KVFullElems, m.KVSWAElems, m.SWAWindow = info.KVFullElems, info.KVSWAElems, info.SWAWindow
	m.StateBytes = info.StateBytes
	m.ActiveParamsB = activeParams(s, *info)
	sort.Slice(m.Variants, func(i, j int) bool { return m.Variants[i].SizeBytes < m.Variants[j].SizeBytes })
	return m, nil
}

// activeParams returns the sources.yaml active_params_b when set (model card
// value), else for MoE models params_b * expert_used_count / expert_count,
// else params_b.
func activeParams(s source, info archInfo) float64 {
	if s.ActiveParamsB != nil {
		return *s.ActiveParamsB
	}
	if info.ExpertCount > 0 && info.ExpertUsed > 0 && info.ExpertUsed < info.ExpertCount {
		// Approximation: shared attention and embedding weights are ignored.
		return round2(s.ParamsB * float64(info.ExpertUsed) / float64(info.ExpertCount))
	}
	return s.ParamsB
}

var errNotFound = errors.New("not found")

func splitTag(tag string) (model, ref string) {
	model, ref, ok := strings.Cut(tag, ":")
	if !ok {
		ref = "latest"
	}
	return model, ref
}

type modelLayer struct {
	Digest string
	Size   uint64
	Extra  uint64 // vision projector layers, loaded with the model
}

const projectorMediaType = "application/vnd.ollama.image.projector"

func fetchManifest(tag string) (modelLayer, error) {
	model, ref := splitTag(tag)
	req, err := http.NewRequest(http.MethodGet, registry+model+"/manifests/"+ref, nil)
	if err != nil {
		return modelLayer{}, err
	}
	req.Header.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")
	req.Header.Set("User-Agent", "gen-catalog")
	resp, err := client.Do(req)
	if err != nil {
		return modelLayer{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return modelLayer{}, errNotFound
	}
	if resp.StatusCode != http.StatusOK {
		return modelLayer{}, fmt.Errorf("manifest: HTTP %s", resp.Status)
	}
	var mf manifest
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&mf); err != nil {
		return modelLayer{}, fmt.Errorf("manifest: %w", err)
	}
	var out modelLayer
	for _, l := range mf.Layers {
		switch l.MediaType {
		case modelMediaType:
			out.Digest, out.Size = l.Digest, l.Size
		case projectorMediaType:
			out.Extra += l.Size
		}
	}
	if out.Digest == "" {
		return modelLayer{}, errors.New("manifest: no model layer")
	}
	return out, nil
}

// fetchArchInfo reads the GGUF header from the start of the blob, retrying
// with a larger range if the first range ends before the needed keys.
func fetchArchInfo(tag, digest string) (archInfo, error) {
	model, _ := splitTag(tag)
	var lastErr error
	for _, n := range []int64{firstRange, maxRange} {
		if n == maxRange {
			log.Printf("range: %s header exceeds %d MiB, refetching %d MiB", tag, firstRange>>20, maxRange>>20)
		}
		data, err := fetchRange(registry+model+"/blobs/"+digest, n)
		if err != nil {
			return archInfo{}, err
		}
		meta, perr := parseGGUF(bytes.NewReader(data))
		info, ierr := archInfoFrom(meta)
		if ierr == nil {
			return info, nil
		}
		if perr == nil || !errors.Is(perr, io.ErrUnexpectedEOF) {
			return archInfo{}, errors.Join(perr, ierr)
		}
		lastErr = errors.Join(perr, ierr)
	}
	return archInfo{}, fmt.Errorf("header larger than %d MiB: %w", maxRange>>20, lastErr)
}

func fetchRange(url string, n int64) ([]byte, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Range", fmt.Sprintf("bytes=0-%d", n-1))
	req.Header.Set("User-Agent", "gen-catalog")
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusPartialContent && resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("blob: HTTP %s", resp.Status)
	}
	// LimitReader protects against servers that ignore Range and send the whole blob.
	return io.ReadAll(io.LimitReader(resp.Body, n))
}

var quantSuffix = regexp.MustCompile(`(?i)-((?:i?q\d[a-z0-9_]*)|fp16|f16|bf16|f32|mxfp4)$`)

// quantFromTag returns the quant suffix of a tag, e.g. "q4_K_M", or "".
func quantFromTag(tag string) string {
	_, ref := splitTag(tag)
	if m := quantSuffix.FindStringSubmatch(ref); m != nil {
		return m[1]
	}
	return ""
}

// quantName names the quant from the tag suffix, else from GGUF general.file_type.
func quantName(tag string, fileType uint64) string {
	if q := quantFromTag(tag); q != "" {
		return q
	}
	if q, ok := fileTypeNames[fileType]; ok {
		return q
	}
	return fmt.Sprintf("ftype%d", fileType)
}

// bits is the average bits per weight, rounded to 2 decimals.
func bits(sizeBytes uint64, paramsB float64) float64 {
	return round2(float64(sizeBytes) * 8 / (paramsB * 1e9))
}

func round2(x float64) float64 { return math.Round(x*100) / 100 }
