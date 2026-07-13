#!/usr/bin/env bash
set -euo pipefail

APP=""
OUTPUT=""
VOLUME_NAME="Noice City"
SIGN_IDENTITY=""

usage() {
  echo "Usage: packaging/macos/build-dmg.sh --app 'Noice City.app' --output Noice-City.dmg [--volume-name NAME] [--sign IDENTITY]"
}

die() {
  echo "build-dmg: $*" >&2
  exit 64
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app) APP="${2:-}"; shift 2 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    --volume-name) VOLUME_NAME="${2:-}"; shift 2 ;;
    --sign) SIGN_IDENTITY="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$(uname -s)" == "Darwin" ]] || die "must run on macOS"
[[ -d "$APP/Contents" ]] || die "invalid app bundle: $APP"
[[ -n "$OUTPUT" ]] || die "--output is required"

mkdir -p "$(dirname "$OUTPUT")"
OUTPUT_DIR="$(cd "$(dirname "$OUTPUT")" && pwd)"
OUTPUT="$OUTPUT_DIR/$(basename "$OUTPUT")"
STAGING="$(mktemp -d "$OUTPUT_DIR/.dmg-staging.XXXXXX")"
trap 'rm -rf "$STAGING"' EXIT

/usr/bin/ditto --noqtn "$APP" "$STAGING/$(basename "$APP")"
/bin/ln -s /Applications "$STAGING/Applications"
/bin/rm -f "$OUTPUT"

/usr/bin/hdiutil create \
  -volname "$VOLUME_NAME" \
  -srcfolder "$STAGING" \
  -format UDZO \
  -imagekey zlib-level=9 \
  -ov \
  "$OUTPUT"

if [[ -n "$SIGN_IDENTITY" ]]; then
  /usr/bin/codesign --force --timestamp --sign "$SIGN_IDENTITY" "$OUTPUT"
  /usr/bin/codesign --verify --strict --verbose=2 "$OUTPUT"
fi

echo "$OUTPUT"
