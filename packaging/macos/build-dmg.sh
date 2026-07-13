#!/usr/bin/env bash
set -euo pipefail

APP=""
OUTPUT=""
VOLUME_NAME="Noice City"
SIGN_IDENTITY=""
RELEASE_MODE=0

usage() {
  echo "Usage: packaging/macos/build-dmg.sh --app 'Noice City.app' --output Noice-City.dmg [--volume-name NAME] [--sign IDENTITY] [--release]"
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
    --release) RELEASE_MODE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$(uname -s)" == "Darwin" ]] || die "must run on macOS"
[[ -d "$APP/Contents" ]] || die "invalid app bundle: $APP"
[[ -n "$OUTPUT" ]] || die "--output is required"

APP="$(cd "$(dirname "$APP")" && pwd)/$(basename "$APP")"
VERIFY_ARGS=("$APP")
if [[ "$RELEASE_MODE" == "1" ]]; then
  VERIFY_ARGS=(--release "$APP")
  [[ -n "$SIGN_IDENTITY" && "$SIGN_IDENTITY" != "-" ]] ||
    die "--release requires a non-ad-hoc --sign identity"
fi
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/verify-app.sh" "${VERIFY_ARGS[@]}"

mkdir -p "$(dirname "$OUTPUT")"
OUTPUT_DIR="$(cd "$(dirname "$OUTPUT")" && pwd)"
OUTPUT="$OUTPUT_DIR/$(basename "$OUTPUT")"
STAGING="$(mktemp -d "$OUTPUT_DIR/.dmg-staging.XXXXXX")"
MOUNT_POINT=""
cleanup() {
  if [[ -n "$MOUNT_POINT" ]] && /sbin/mount | /usr/bin/grep -Fq " on $MOUNT_POINT "; then
    /usr/bin/hdiutil detach "$MOUNT_POINT" >/dev/null 2>&1 || true
  fi
  [[ -z "$MOUNT_POINT" ]] || /bin/rm -rf "$MOUNT_POINT"
  /bin/rm -rf "$STAGING"
}
trap cleanup EXIT

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

# Verify the copy actually stored inside the finished disk image. This catches
# staging, image-creation, and path-handling defects before notarization.
MOUNT_POINT="$(mktemp -d "${TMPDIR:-/tmp}/noice-city-dmg.XXXXXX")"
/usr/bin/hdiutil attach -readonly -nobrowse -mountpoint "$MOUNT_POINT" "$OUTPUT" >/dev/null
MOUNTED_APP="$MOUNT_POINT/$(basename "$APP")"
[[ -d "$MOUNTED_APP/Contents" ]] || die "finished DMG does not contain $(basename "$APP")"
MOUNT_VERIFY_ARGS=("$MOUNTED_APP")
if [[ "$RELEASE_MODE" == "1" ]]; then
  MOUNT_VERIFY_ARGS=(--release "$MOUNTED_APP")
fi
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/verify-app.sh" "${MOUNT_VERIFY_ARGS[@]}"
/usr/bin/hdiutil detach "$MOUNT_POINT" >/dev/null
/bin/rm -rf "$MOUNT_POINT"
MOUNT_POINT=""

echo "$OUTPUT"
