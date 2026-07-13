#!/usr/bin/env bash
set -euo pipefail

APP="${1:-}"
if [[ ! -d "$APP/Contents" ]]; then
  echo "Usage: packaging/macos/verify-app.sh 'Noice City.app'" >&2
  exit 64
fi

PLIST="$APP/Contents/Info.plist"
RUNTIME="$APP/Contents/Resources/runtime/bin/python3"
APP_ROOT="$APP/Contents/Resources/app"

/usr/bin/plutil -lint "$PLIST"
[[ -x "$APP/Contents/MacOS/NoiceCity" ]]
[[ "$(/usr/bin/file -b "$APP/Contents/MacOS/NoiceCity")" == *Mach-O* ]]
[[ -x "$APP/Contents/Resources/NoiceCity.launcher" ]]
[[ -x "$RUNTIME" ]]
[[ -d "$APP_ROOT/voice_trainer" ]]
[[ -d "$APP_ROOT/static" ]]
[[ -d "$APP_ROOT/assets" ]]

RUNTIME_MACHINE="$("$RUNTIME" -B -c 'import platform; print(platform.machine())')"
RUNTIME_VERSION="$("$RUNTIME" -B -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$RUNTIME_MACHINE" == "arm64" ]] || {
  echo "Bundled runtime is $RUNTIME_MACHINE, expected arm64" >&2
  exit 1
}
[[ "$RUNTIME_VERSION" == "3.12" ]] || {
  echo "Bundled runtime is Python $RUNTIME_VERSION, expected 3.12" >&2
  exit 1
}

for forbidden in .venv state.sqlite session-token recordings datasets checkpoints models; do
  if /usr/bin/find "$APP_ROOT" -name "$forbidden" -print -quit | /usr/bin/grep -q .; then
    echo "Forbidden artifact in app: $forbidden" >&2
    exit 1
  fi
done

if /usr/bin/codesign -d "$APP" >/dev/null 2>&1; then
  while IFS= read -r -d '' candidate; do
    if /usr/bin/file -b "$candidate" | /usr/bin/grep -q 'Mach-O'; then
      /usr/bin/codesign --verify --strict --verbose=2 "$candidate"
    fi
  done < <(/usr/bin/find "$APP/Contents" -type f -print0)
  /usr/bin/codesign --verify --strict --verbose=2 "$APP"
  /usr/bin/codesign --verify --deep --strict --verbose=2 "$APP"
fi

echo "Verified: $APP"
