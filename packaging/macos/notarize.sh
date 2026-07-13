#!/usr/bin/env bash
set -euo pipefail

DMG=""
KEYCHAIN_PROFILE=""

usage() {
  cat <<'EOF'
Usage: packaging/macos/notarize.sh --dmg Noice-City.dmg --keychain-profile PROFILE

Create PROFILE once with:
  xcrun notarytool store-credentials PROFILE --apple-id you@example.com --team-id TEAMID1234

Enter the app-specific password at the secure prompt. Do not place it directly
on the command line or in shell history.
EOF
}

die() {
  echo "notarize: $*" >&2
  exit 64
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dmg) DMG="${2:-}"; shift 2 ;;
    --keychain-profile) KEYCHAIN_PROFILE="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -f "$DMG" ]] || die "DMG not found: $DMG"
[[ -n "$KEYCHAIN_PROFILE" ]] || die "--keychain-profile is required"

/usr/bin/codesign --verify --strict --verbose=2 "$DMG"

RESULT_FILE="$DMG.notary.json"
LOG_FILE="$DMG.notary-log.json"
set +e
/usr/bin/xcrun notarytool submit "$DMG" \
  --keychain-profile "$KEYCHAIN_PROFILE" \
  --wait \
  --output-format json > "$RESULT_FILE"
SUBMIT_RC=$?
set -e

/bin/cat "$RESULT_FILE"
SUBMISSION_ID="$(/usr/bin/plutil -extract id raw -o - "$RESULT_FILE" 2>/dev/null || true)"
STATUS="$(/usr/bin/plutil -extract status raw -o - "$RESULT_FILE" 2>/dev/null || true)"

if [[ "$SUBMIT_RC" -ne 0 || "$STATUS" != "Accepted" ]]; then
  if [[ -n "$SUBMISSION_ID" ]]; then
    /usr/bin/xcrun notarytool log "$SUBMISSION_ID" \
      --keychain-profile "$KEYCHAIN_PROFILE" \
      "$LOG_FILE" || true
    [[ ! -f "$LOG_FILE" ]] || /bin/cat "$LOG_FILE"
  fi
  echo "notarize: submission was not accepted (status=${STATUS:-unknown})" >&2
  exit 1
fi

/usr/bin/xcrun stapler staple "$DMG"
/usr/bin/xcrun stapler validate "$DMG"
/usr/bin/codesign --verify --strict --verbose=2 "$DMG"
/usr/sbin/spctl --assess --type open --context context:primary-signature -v "$DMG"
