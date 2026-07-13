#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

APP_NAME="Noice City"
BUNDLE_ID="com.ruhitaj.noicecity"
VERSION="0.1.0"
BUILD_NUMBER="2"
RUNTIME_DIR=""
OUTPUT_DIR="$PROJECT_ROOT/dist/macos"
DEFAULT_ICON_FILE="$PROJECT_ROOT/assets/branding/NoiceCity.icns"
ICON_FILE="$DEFAULT_ICON_FILE"
ICON_EXPLICIT=0
SIGN_IDENTITY=""

usage() {
  cat <<'EOF'
Build the Apple Silicon Noice City.app bundle.

Usage:
  packaging/macos/build-app.sh --runtime DIR [options]

Required:
  --runtime DIR       Relocatable arm64 Python 3.12 runtime (bin/python3).

Options:
  --output DIR        Destination directory (default: dist/macos).
  --name NAME         Product name (default: Noice City).
  --bundle-id ID      Reverse-DNS bundle identifier.
  --version VERSION   Marketing version.
  --build NUMBER      Integer build number.
  --icon FILE.icns    Override the default assets/branding/NoiceCity.icns icon.
  --sign IDENTITY     Codesign identity; use '-' for ad-hoc signing.
  -h, --help          Show this help.

This builder uses an explicit source allowlist. It never copies .venv, models,
recordings, datasets, checkpoints, caches, state.sqlite, tokens, or user data.
EOF
}

die() {
  echo "build-app: $*" >&2
  exit 64
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime) RUNTIME_DIR="${2:-}"; shift 2 ;;
    --output) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --name) APP_NAME="${2:-}"; shift 2 ;;
    --bundle-id) BUNDLE_ID="${2:-}"; shift 2 ;;
    --version) VERSION="${2:-}"; shift 2 ;;
    --build) BUILD_NUMBER="${2:-}"; shift 2 ;;
    --icon) ICON_FILE="${2:-}"; ICON_EXPLICIT=1; shift 2 ;;
    --sign) SIGN_IDENTITY="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$(uname -s)" == "Darwin" ]] || die "must run on macOS"
[[ "$(uname -m)" == "arm64" ]] || die "must run on Apple Silicon"
[[ -x /usr/bin/xcrun ]] || die "Xcode command-line tools are required"
[[ -n "$RUNTIME_DIR" ]] || die "--runtime is required"
[[ -x "$RUNTIME_DIR/bin/python3" ]] || die "$RUNTIME_DIR/bin/python3 is missing"
[[ "$APP_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._[:space:]-]*$ ]] || die "invalid app name"
[[ "$BUNDLE_ID" =~ ^[A-Za-z0-9.-]+$ ]] || die "invalid bundle id"
[[ "$VERSION" =~ ^[0-9]+([.][0-9]+){1,3}$ ]] || die "invalid version"
[[ "$BUILD_NUMBER" =~ ^[0-9]+$ ]] || die "build number must be an integer"
if [[ "$ICON_EXPLICIT" == "1" ]]; then
  [[ -f "$ICON_FILE" ]] || die "icon not found: $ICON_FILE"
  [[ "$ICON_FILE" == *.icns ]] || die "icon must be an .icns file"
elif [[ ! -f "$ICON_FILE" ]]; then
  ICON_FILE=""
  echo "build-app: default icon not found at $DEFAULT_ICON_FILE; Finder will use the generic app icon" >&2
fi

RUNTIME_DIR="$(cd "$RUNTIME_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
APP_BUNDLE="$OUTPUT_DIR/$APP_NAME.app"
STAGING="$(mktemp -d "$OUTPUT_DIR/.app-staging.XXXXXX")"
trap 'rm -rf "$STAGING"' EXIT

CONTENTS="$STAGING/$APP_NAME.app/Contents"
RESOURCES="$CONTENTS/Resources"
APP_RESOURCES="$RESOURCES/app"
mkdir -p "$CONTENTS/MacOS" "$APP_RESOURCES"

# Copy only Git-tracked application files. This prevents an ignored recording,
# credential, cache, or other local file from entering a release merely because
# it was placed below an otherwise allowlisted source directory.
command -v git >/dev/null 2>&1 || die "git is required to enumerate release sources"
git -C "$PROJECT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
  die "project root is not a Git working tree"
TRACKED_SOURCE_COUNT=0
while IFS= read -r -d '' relative; do
  case "$relative" in
    voice_trainer/*|static/*) ;;
    *) die "Git returned an unexpected release source: $relative" ;;
  esac
  source_file="$PROJECT_ROOT/$relative"
  [[ -f "$source_file" && ! -L "$source_file" ]] ||
    die "tracked release source is missing, not regular, or a symlink: $relative"
  destination="$APP_RESOURCES/$relative"
  mkdir -p "$(dirname "$destination")"
  /bin/cp "$source_file" "$destination"
  TRACKED_SOURCE_COUNT=$((TRACKED_SOURCE_COUNT + 1))
done < <(git -C "$PROJECT_ROOT" ls-files -z -- voice_trainer static)
[[ "$TRACKED_SOURCE_COUNT" -gt 10 ]] || die "tracked application source set is unexpectedly small"

for relative in assets/prompts.json pyproject.toml; do
  git -C "$PROJECT_ROOT" ls-files --error-unmatch -- "$relative" >/dev/null 2>&1 ||
    die "required release source is not Git-tracked: $relative"
  [[ -f "$PROJECT_ROOT/$relative" && ! -L "$PROJECT_ROOT/$relative" ]] ||
    die "required release source is missing, not regular, or a symlink: $relative"
  mkdir -p "$APP_RESOURCES/$(dirname "$relative")"
  /bin/cp "$PROJECT_ROOT/$relative" "$APP_RESOURCES/$relative"
done
for notice_file in LICENSE NOTICE TRADEMARKS.md THIRD_PARTY_NOTICES.md MODEL_NOTICES.md; do
  [[ -f "$PROJECT_ROOT/$notice_file" ]] || die "required release notice is missing: $notice_file"
  /bin/cp "$PROJECT_ROOT/$notice_file" "$RESOURCES/$notice_file"
done

# The runtime must be prepared separately from a relocatable Python
# distribution. A project .venv is intentionally rejected by policy.
case "$RUNTIME_DIR" in
  "$PROJECT_ROOT/.venv"|"$PROJECT_ROOT/.venv"/*)
    die "a .venv is not relocatable; use prepare-runtime.sh"
    ;;
esac
/usr/bin/ditto --noqtn "$RUNTIME_DIR" "$RESOURCES/runtime"

# prepare-runtime.sh obtains these notices and provenance from the exact pinned
# full python-build-standalone companion archive. Promote them to an obvious
# top-level Resources location rather than burying them inside the interpreter.
RUNTIME_RELEASE="$RESOURCES/runtime/share/noice-city/release/python-build-standalone"
DEPENDENCY_RELEASE="$RESOURCES/runtime/share/noice-city/release/runtime-dependencies"
RUNTIME_PACKAGE_LICENSE_RELEASE="$RESOURCES/runtime/share/noice-city/release/runtime-package-licenses"
for provenance_file in \
  PYTHON.json ARCHIVE_SHA256S.txt LICENSES_SHA256.txt SUPPLEMENTAL_LICENSE_PROVENANCE.txt; do
  [[ -s "$RUNTIME_RELEASE/$provenance_file" ]] || die "runtime release provenance is missing: $provenance_file"
done
if ! /usr/bin/find "$RUNTIME_RELEASE/licenses" -type f -print -quit | /usr/bin/grep -q .; then
  die "runtime release provenance contains no Python component licenses"
fi
for provenance_file in \
  uv.lock requirements.txt INSTALLED_PACKAGES.txt PIP_CHECK.txt \
  MLX_AUDIO_METADATA_OVERLAY.json RUNTIME_PACKAGE_LICENSES_SHA256.txt \
  RUNTIME_PACKAGE_LICENSE_PROVENANCE.txt SHA256SUMS.txt; do
  [[ -s "$DEPENDENCY_RELEASE/$provenance_file" ]] || \
    die "runtime dependency provenance is missing: $provenance_file"
done
if ! /usr/bin/find "$RUNTIME_PACKAGE_LICENSE_RELEASE" -type f -print -quit | /usr/bin/grep -q .; then
  die "runtime package license set is empty"
fi
mkdir -p \
  "$RESOURCES/THIRD_PARTY_LICENSES/python-build-standalone" \
  "$RESOURCES/THIRD_PARTY_LICENSES/runtime-packages" \
  "$RESOURCES/RELEASE_PROVENANCE/python-build-standalone" \
  "$RESOURCES/RELEASE_PROVENANCE/runtime-dependencies"
/usr/bin/ditto --noqtn \
  "$RUNTIME_RELEASE/licenses" \
  "$RESOURCES/THIRD_PARTY_LICENSES/python-build-standalone"
/usr/bin/ditto --noqtn \
  "$RUNTIME_PACKAGE_LICENSE_RELEASE" \
  "$RESOURCES/THIRD_PARTY_LICENSES/runtime-packages"
/usr/bin/ditto --noqtn \
  "$DEPENDENCY_RELEASE" \
  "$RESOURCES/RELEASE_PROVENANCE/runtime-dependencies"
for provenance_file in \
  PYTHON.json ARCHIVE_SHA256S.txt LICENSES_SHA256.txt SUPPLEMENTAL_LICENSE_PROVENANCE.txt; do
  /bin/cp \
    "$RUNTIME_RELEASE/$provenance_file" \
    "$RESOURCES/RELEASE_PROVENANCE/python-build-standalone/$provenance_file"
done
/bin/rm -rf "$RESOURCES/runtime/share/noice-city/release"

/bin/cp "$SCRIPT_DIR/templates/Info.plist" "$CONTENTS/Info.plist"
/bin/cp "$SCRIPT_DIR/templates/NoiceCity.launcher" "$RESOURCES/NoiceCity.launcher"
/bin/chmod 755 "$RESOURCES/NoiceCity.launcher"
/usr/bin/xcrun --sdk macosx clang \
  -arch arm64 \
  -mmacosx-version-min=26.2 \
  -Os -Wall -Wextra -Werror \
  "$SCRIPT_DIR/templates/NoiceCityLauncher.c" \
  -o "$CONTENTS/MacOS/NoiceCity"

PLIST_BUDDY=/usr/libexec/PlistBuddy
"$PLIST_BUDDY" -c "Set :CFBundleDisplayName $APP_NAME" "$CONTENTS/Info.plist"
"$PLIST_BUDDY" -c "Set :CFBundleName $APP_NAME" "$CONTENTS/Info.plist"
"$PLIST_BUDDY" -c "Set :CFBundleIdentifier $BUNDLE_ID" "$CONTENTS/Info.plist"
"$PLIST_BUDDY" -c "Set :CFBundleShortVersionString $VERSION" "$CONTENTS/Info.plist"
"$PLIST_BUDDY" -c "Set :CFBundleVersion $BUILD_NUMBER" "$CONTENTS/Info.plist"

if [[ -n "$ICON_FILE" ]]; then
  /bin/cp "$ICON_FILE" "$RESOURCES/AppIcon.icns"
  "$PLIST_BUDDY" -c "Add :CFBundleIconFile string AppIcon" "$CONTENTS/Info.plist"
fi

# Never ship bytecode, test debris, mutable data, or VCS metadata even if one
# accidentally appears in an allowlisted source directory later.
/usr/bin/find "$STAGING/$APP_NAME.app" \
  \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.git' \) \
  -type d -prune -exec /bin/rm -rf {} +
/usr/bin/find "$STAGING/$APP_NAME.app" \
  \( -name '*.pyc' -o -name '*.pyo' -o -name '.DS_Store' \) \
  -type f -delete

for forbidden in .venv state.sqlite session-token manifest.json recordings datasets checkpoints models; do
  if /usr/bin/find "$APP_RESOURCES" -name "$forbidden" -print -quit | /usr/bin/grep -q .; then
    die "forbidden private/build artifact entered bundle: $forbidden"
  fi
done

if /usr/bin/find "$APP_RESOURCES" -type f \
  \( -iname '*.pem' -o -iname '*.key' -o -iname '*.p12' -o -iname '*.pfx' \
     -o -iname '*.cer' -o -iname '*.crt' -o -iname '*.mobileprovision' \) \
  -print -quit | /usr/bin/grep -q .; then
  die "credential or signing material entered the application source bundle"
fi

if /usr/bin/find "$APP_RESOURCES/assets" -mindepth 1 \
  ! -path "$APP_RESOURCES/assets/prompts.json" -print -quit | /usr/bin/grep -q .; then
  die "an unallowlisted source asset entered the application bundle"
fi

RUNTIME_MACHINE="$("$RESOURCES/runtime/bin/python3" -B -c 'import platform; print(platform.machine())')"
[[ "$RUNTIME_MACHINE" == "arm64" ]] || die "runtime is $RUNTIME_MACHINE, expected arm64"
RUNTIME_VERSION="$("$RESOURCES/runtime/bin/python3" -B -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$RUNTIME_VERSION" == "3.12" ]] || die "runtime is Python $RUNTIME_VERSION, expected 3.12"

if /usr/bin/find "$STAGING/$APP_NAME.app" -type d -name '__pycache__' -print -quit | /usr/bin/grep -q .; then
  die "runtime validation wrote bytecode into the staged bundle"
fi

/bin/rm -rf "$APP_BUNDLE"
/bin/mv "$STAGING/$APP_NAME.app" "$APP_BUNDLE"

if [[ -n "$SIGN_IDENTITY" ]]; then
  "$SCRIPT_DIR/sign-app.sh" --app "$APP_BUNDLE" --identity "$SIGN_IDENTITY"
fi

echo "$APP_BUNDLE"
