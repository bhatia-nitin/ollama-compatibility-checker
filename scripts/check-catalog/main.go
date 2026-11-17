// Command check-catalog reports how catalog/sources.yaml has drifted from the
// public Ollama library. It never edits the catalog: quality scores need a
// human, so it only writes a markdown report.
//
//	go run ./scripts/check-catalog -out report.md
//
// It reports (1) catalog tags whose registry manifest now 404s, (2) new quant
// tags in tracked families that match the catalog quant policy, and (3)
// library models whose family is not tracked and not in catalog/check-ignore.txt.
//
// The ollama.com pages are scraped from HTML. A page that parses to zero
// entries is an error ("page format changed"), never "nothing new". Requests
// are sequential and identify themselves with a generic User-Agent. Exit
// status is 0 whenever a report was produced, non-zero on fetch/parse failure.
package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

const (
	registryBase = "https://registry.ollama.ai/v2/library/"
	libraryURL   = "https://ollama.com/library"
	userAgent    = "ollama-compat-catalog-check"
	perFamilyCap = 30
	maxBody      = 8 << 20
)

var client = &http.Client{Timeout: 30 * time.Second}

type source struct {
	Family string   `yaml:"family"`
	Tags   []string `yaml:"tags"`
}

var (
	libraryRe = regexp.MustCompile(`href="/library/([a-zA-Z0-9._-]+)"`)
	quantRe   = regexp.MustCompile(`^.+-(q4_K_M|q8_0|q3_K_M)$`)
)

// parseLibrary returns the sorted, de-duplicated model names linked from the
// library page. Zero names is an error.
func parseLibrary(html string) ([]string, error) {
	seen := map[string]bool{}
	for _, m := range libraryRe.FindAllStringSubmatch(html, -1) {
		seen[m[1]] = true
	}
	if len(seen) == 0 {
		return nil, errors.New("library page: no models found (page format changed)")
	}
	return sortedKeys(seen), nil
}

// parseTags returns the sorted, de-duplicated tag names (without the
// "model:" prefix) found on a model's tags page. Zero tags is an error.
func parseTags(model, html string) ([]string, error) {
	re := regexp.MustCompile(`(?:^|[^a-zA-Z0-9._-])` + regexp.QuoteMeta(model) + `:([a-zA-Z0-9._-]+)`)
	seen := map[string]bool{}
	for _, m := range re.FindAllStringSubmatch(html, -1) {
		seen[m[1]] = true
	}
	if len(seen) == 0 {
		return nil, fmt.Errorf("%s tags page: no tags found (page format changed)", model)
	}
	return sortedKeys(seen), nil
}

// quantPolicy reports whether a tag name matches the catalog quant policy: a
// size-qualified tag ending in q4_K_M, q8_0 or q3_K_M. fp16, bare size tags
// and "latest" do not match.
func quantPolicy(tag string) bool { return quantRe.MatchString(tag) }

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// parseIgnore reads the ignore file: one name per line, # starts a comment.
func parseIgnore(r io.Reader) map[string]bool {
	out := map[string]bool{}
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		line := sc.Text()
		if i := strings.Index(line, "#"); i >= 0 {
			line = line[:i]
		}
		if line = strings.TrimSpace(line); line != "" {
			out[line] = true
		}
	}
	return out
}

// splitTag splits "family:tag".
func splitTag(full string) (family, tag string, ok bool) {
	family, tag, ok = strings.Cut(full, ":")
	return family, tag, ok && family != "" && tag != ""
}

type report struct {
	Missing     []string            // "family:tag" whose manifest 404s
	NewTags     map[string][]string // family -> full tags (uncapped)
	NewFamilies []string
	Families    int
	TagsChecked int
}

// newQuantTags returns policy-matching tags from the page that the catalog
// does not already list. have is the set of catalog tags (full "family:tag").
func newQuantTags(family string, pageTags []string, have map[string]bool) []string {
	var out []string
	for _, t := range pageTags {
		full := family + ":" + t
		if quantPolicy(t) && !have[full] {
			out = append(out, full)
		}
	}
	return out
}

// newFamilies returns library names that are neither tracked nor ignored.
func newFamilies(lib []string, tracked, ignore map[string]bool) []string {
	var out []string
	for _, n := range lib {
		if !tracked[n] && !ignore[n] {
			out = append(out, n)
		}
	}
	return out
}

func (r report) hasFindings() bool {
	return len(r.Missing) > 0 || len(r.NewTags) > 0 || len(r.NewFamilies) > 0
}

func (r report) markdown() string {
	var b strings.Builder
	b.WriteString("# Catalog check: Ollama library changes\n\n")
	fmt.Fprintf(&b, "Checked %d tracked tags across %d families against the Ollama library on %s.\n\n",
		r.TagsChecked, r.Families, time.Now().UTC().Format("2006-01-02"))
	if !r.hasFindings() {
		b.WriteString("No drift found.\n")
		return b.String()
	}
	b.WriteString("## Missing tags\n\n")
	if len(r.Missing) == 0 {
		b.WriteString("None.\n\n")
	} else {
		b.WriteString("Tags in `catalog/sources.yaml` whose registry manifest now returns 404 (a catalog regen would fail):\n\n")
		for _, t := range r.Missing {
			fmt.Fprintf(&b, "- `%s`\n", t)
		}
		b.WriteString("\n")
	}
	b.WriteString("## New quants and sizes in tracked families\n\n")
	if len(r.NewTags) == 0 {
		b.WriteString("None.\n\n")
	} else {
		b.WriteString("Tags matching the quant policy (q4_K_M, q8_0, q3_K_M) that are not in `catalog/sources.yaml`:\n\n")
		fams := make([]string, 0, len(r.NewTags))
		for f := range r.NewTags {
			fams = append(fams, f)
		}
		sort.Strings(fams)
		for _, f := range fams {
			tags := r.NewTags[f]
			fmt.Fprintf(&b, "### %s (%d)\n\n", f, len(tags))
			for i, t := range tags {
				if i == perFamilyCap {
					fmt.Fprintf(&b, "- ... and %d more\n", len(tags)-perFamilyCap)
					break
				}
				fmt.Fprintf(&b, "- `%s`\n", t)
			}
			b.WriteString("\n")
		}
	}
	b.WriteString("## New families\n\n")
	if len(r.NewFamilies) == 0 {
		b.WriteString("None.\n")
	} else {
		b.WriteString("Library models not in `catalog/sources.yaml` and not in `catalog/check-ignore.txt`. Add the ones to skip to the ignore file:\n\n")
		for _, n := range r.NewFamilies {
			fmt.Fprintf(&b, "- [%s](https://ollama.com/library/%s)\n", n, n)
		}
	}
	return b.String()
}

func get(url, accept string) (*http.Response, error) {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", userAgent)
	if accept != "" {
		req.Header.Set("Accept", accept)
	}
	return client.Do(req)
}

func fetchPage(url string) (string, error) {
	resp, err := get(url, "")
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("%s: HTTP %d", url, resp.StatusCode)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, maxBody))
	return string(b), err
}

// manifestExists returns false on 404, an error on anything unexpected.
func manifestExists(family, tag string) (bool, error) {
	resp, err := get(registryBase+family+"/manifests/"+tag, "application/vnd.docker.distribution.manifest.v2+json")
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
	switch resp.StatusCode {
	case http.StatusOK:
		return true, nil
	case http.StatusNotFound:
		return false, nil
	}
	return false, fmt.Errorf("manifest %s:%s: HTTP %d", family, tag, resp.StatusCode)
}

func run(in, ignorePath, out string) error {
	raw, err := os.ReadFile(in)
	if err != nil {
		return err
	}
	var srcs []source
	if err := yaml.Unmarshal(raw, &srcs); err != nil {
		return fmt.Errorf("%s: %w", in, err)
	}
	f, err := os.Open(ignorePath)
	if err != nil {
		return err
	}
	ignore := parseIgnore(f)
	f.Close()

	tracked := map[string]bool{}
	have := map[string]bool{}
	var allTags []string
	for _, s := range srcs {
		tracked[s.Family] = true
		for _, t := range s.Tags {
			if !have[t] {
				have[t] = true
				allTags = append(allTags, t)
			}
		}
	}
	if len(tracked) == 0 || len(allTags) == 0 {
		return fmt.Errorf("%s: no families or tags", in)
	}

	rep := report{NewTags: map[string][]string{}, Families: len(tracked), TagsChecked: len(allTags)}

	for _, full := range allTags {
		fam, tag, ok := splitTag(full)
		if !ok {
			return fmt.Errorf("bad tag %q in %s", full, in)
		}
		exists, err := manifestExists(fam, tag)
		if err != nil {
			return err
		}
		if !exists {
			rep.Missing = append(rep.Missing, full)
		}
	}

	for _, fam := range sortedKeys(tracked) {
		page, err := fetchPage(libraryURL + "/" + fam + "/tags")
		if err != nil {
			return err
		}
		tags, err := parseTags(fam, page)
		if err != nil {
			return err
		}
		if n := newQuantTags(fam, tags, have); len(n) > 0 {
			rep.NewTags[fam] = n
		}
	}

	page, err := fetchPage(libraryURL)
	if err != nil {
		return err
	}
	lib, err := parseLibrary(page)
	if err != nil {
		return err
	}
	rep.NewFamilies = newFamilies(lib, tracked, ignore)

	nt := 0
	for _, v := range rep.NewTags {
		nt += len(v)
	}
	log.Printf("checked %d tags in %d families: %d missing, %d new tags in %d families, %d new families",
		rep.TagsChecked, rep.Families, len(rep.Missing), nt, len(rep.NewTags), len(rep.NewFamilies))
	return os.WriteFile(out, []byte(rep.markdown()), 0o644)
}

func main() {
	in := flag.String("in", "catalog/sources.yaml", "catalog source list")
	ignore := flag.String("ignore", "catalog/check-ignore.txt", "ignore list for new-family check")
	out := flag.String("out", "report.md", "markdown report path")
	flag.Parse()
	log.SetFlags(0)
	if err := run(*in, *ignore, *out); err != nil {
		log.Fatal(err)
	}
}
