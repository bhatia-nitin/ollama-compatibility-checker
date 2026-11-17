// Command add-calibration turns the body of a calibration issue into a record
// file under calibration/data. The body is untrusted: it is only parsed as
// JSON into calib.Record (unknown fields rejected), checked, and written back
// from the struct, so nothing but known fields reaches the repo.
//
//	go run ./scripts/add-calibration -body body.md -issue 42
//
// It prints the path it wrote. CI's calibration workflow then runs
// scripts/fit-calibration and opens a pull request.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
)

func main() {
	body := flag.String("body", "", "file holding the issue body")
	issue := flag.Int("issue", 0, "issue number, used in the file name")
	dir := flag.String("dir", "calibration/data", "output directory")
	flag.Parse()
	log.SetFlags(0)
	if *body == "" || *issue <= 0 {
		log.Fatal("usage: add-calibration -body FILE -issue N")
	}
	raw, err := os.ReadFile(*body)
	if err != nil {
		log.Fatal(err)
	}
	r, err := parseIssue(raw)
	if err != nil {
		log.Fatal(err)
	}
	js, err := encode(r)
	if err != nil {
		log.Fatal(err)
	}
	path := filepath.Join(*dir, fileName(r, *issue))
	if err := os.WriteFile(path, js, 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Println(path)
}
