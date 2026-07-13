#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP=""
IDENTITY=""

usage() {
  echo "Usage: packaging/macos/sign-app.sh --app 'Noice City.app' --identity IDENTITY"
}

die() {
  echo "sign-app: $*" >&2
  exit 64
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app) APP="${2:-}"; shift 2 ;;
    --identity) IDENTITY="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -d "$APP/Contents" ]] || die "invalid app bundle: $APP"
[[ -n "$IDENTITY" ]] || die "--identity is required"

SIGN_FLAGS=(--force --sign "$IDENTITY")
if [[ "$IDENTITY" != "-" ]]; then
  SIGN_FLAGS+=(--options runtime --timestamp)
fi

# Finder metadata, quarantine flags, and downloaded-file provenance must not
# enter the sealed bundle as unsigned extended attributes.
/usr/bin/xattr -cr "$APP"

# Sign every nested Mach-O object first; then sign the launcher and outer app.
# This avoids --deep, which can hide an incomplete or incorrectly ordered seal.
while IFS= read -r -d '' candidate; do
  if /usr/bin/file -b "$candidate" | /usr/bin/grep -q 'Mach-O'; then
    /usr/bin/codesign "${SIGN_FLAGS[@]}" "$candidate"
  fi
done < <(/usr/bin/find "$APP/Contents/Resources/runtime" -type f -print0)

/usr/bin/codesign "${SIGN_FLAGS[@]}" "$APP/Contents/MacOS/NoiceCity"
/usr/bin/codesign "${SIGN_FLAGS[@]}" "$APP"
/usr/bin/codesign --verify --strict --verbose=2 "$APP"
/usr/bin/codesign --verify --deep --strict --verbose=2 "$APP"
