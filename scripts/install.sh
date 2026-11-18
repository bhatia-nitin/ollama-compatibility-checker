#!/bin/sh
# Install ollama-compat from GitHub Releases (macOS and Linux).
#
#   curl -fsSL https://raw.githubusercontent.com/shra26/ollama-compatibility-checker/main/scripts/install.sh | sh
#
# Environment:
#   VERSION      release to install, e.g. v0.1.1 or 0.1.1 (default: latest)
#   INSTALL_DIR  destination directory (default: $HOME/.local/bin)
#
# Downloads only the release archive and checksums.txt from github.com,
# verifies the SHA-256 before extracting, and sends no telemetry.
set -eu

REPO="shra26/ollama-compatibility-checker"
BIN="ollama-compat"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"

err() { printf 'error: %s\n' "$*" >&2; exit 1; }
info() { printf '%s\n' "$*"; }

fetch() { # fetch URL DEST
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 -o "$2" "$1"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$2" "$1"
  else
    err "need curl or wget"
  fi
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d ' ' -f 1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d ' ' -f 1
  elif command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha256 "$1" | sed 's/^.*= //'
  else
    err "need sha256sum, shasum or openssl to verify the download"
  fi
}

case "$(uname -s)" in
  Linux)  os=linux ;;
  Darwin) os=darwin ;;
  *) err "unsupported OS: $(uname -s) (on Windows use scripts/install.ps1)" ;;
esac
case "$(uname -m)" in
  x86_64|amd64)  arch=amd64 ;;
  arm64|aarch64) arch=arm64 ;;
  *) err "unsupported architecture: $(uname -m)" ;;
esac

TMP="$(mktemp -d 2>/dev/null || mktemp -d -t ollama-compat)"
trap 'rm -rf "$TMP"' EXIT INT TERM

tag="${VERSION:-}"
if [ -z "$tag" ]; then
  fetch "https://api.github.com/repos/$REPO/releases/latest" "$TMP/latest.json"
  tag="$(sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$TMP/latest.json" | head -n 1)"
  [ -n "$tag" ] || err "could not determine the latest release (GitHub API rate limit? set VERSION=vX.Y.Z)"
fi
case "$tag" in v*) ;; *) tag="v$tag" ;; esac
version="${tag#v}"

archive="${BIN}_${version}_${os}_${arch}.tar.gz"
base="https://github.com/$REPO/releases/download/$tag"

info "Installing $BIN $tag ($os/$arch)"
fetch "$base/$archive" "$TMP/$archive" || err "download failed: $base/$archive"
fetch "$base/checksums.txt" "$TMP/checksums.txt" || err "download failed: $base/checksums.txt"

expected="$(awk -v f="$archive" '$2 == f || $2 == "*" f { print $1; exit }' "$TMP/checksums.txt")"
[ -n "$expected" ] || err "no checksum listed for $archive"
actual="$(sha256_of "$TMP/$archive")"
if [ "$expected" != "$actual" ]; then
  err "checksum mismatch for $archive (expected $expected, got $actual); nothing installed"
fi
info "Checksum OK"

mkdir "$TMP/x"
tar -xzf "$TMP/$archive" -C "$TMP/x" "$BIN" || err "could not extract $BIN from archive"

mkdir -p "$INSTALL_DIR" || err "cannot create $INSTALL_DIR (set INSTALL_DIR to a writable directory)"
cp "$TMP/x/$BIN" "$INSTALL_DIR/$BIN.new" || err "cannot write to $INSTALL_DIR"
chmod 755 "$INSTALL_DIR/$BIN.new"
mv -f "$INSTALL_DIR/$BIN.new" "$INSTALL_DIR/$BIN"

if [ "$os" = darwin ]; then
  xattr -d com.apple.quarantine "$INSTALL_DIR/$BIN" 2>/dev/null || true
fi

info "Installed $INSTALL_DIR/$BIN"
case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *) info "Note: $INSTALL_DIR is not on your PATH. Add it, e.g.:"
     info "  export PATH=\"$INSTALL_DIR:\$PATH\"" ;;
esac
