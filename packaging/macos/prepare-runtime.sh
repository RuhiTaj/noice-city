#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_ARCHIVE=""
OUTPUT_DIR="$PROJECT_ROOT/build/macos/runtime"

usage() {
  cat <<'EOF'
Prepare a relocatable Apple Silicon Python runtime for the .app bundle.

Usage:
  packaging/macos/prepare-runtime.sh --python-archive FILE.tar.zst [--output DIR]

The archive must be a trusted, pinned Python 3.12 arm64 "install_only"
python-build-standalone artifact containing python/bin/python3. This command is
a release-build step and may resolve the locked Python dependencies. It never
downloads application models or copies the project's .venv.
EOF
}

die() {
  echo "prepare-runtime: $*" >&2
  exit 64
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-archive) PYTHON_ARCHIVE="${2:-}"; shift 2 ;;
    --output) OUTPUT_DIR="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$(uname -s)" == "Darwin" ]] || die "must run on macOS"
[[ "$(uname -m)" == "arm64" ]] || die "must run on Apple Silicon"
[[ -n "$PYTHON_ARCHIVE" ]] || die "--python-archive is required"
[[ -f "$PYTHON_ARCHIVE" ]] || die "archive not found: $PYTHON_ARCHIVE"
command -v uv >/dev/null 2>&1 || die "uv is required for a locked dependency export"

mkdir -p "$(dirname "$OUTPUT_DIR")"
OUTPUT_PARENT="$(cd "$(dirname "$OUTPUT_DIR")" && pwd)"
STAGING="$(mktemp -d "$OUTPUT_PARENT/.runtime-staging.XXXXXX")"
trap 'rm -rf "$STAGING"' EXIT

/usr/bin/ditto -x -k "$PYTHON_ARCHIVE" "$STAGING/unpacked" 2>/dev/null || {
  mkdir -p "$STAGING/unpacked"
  /usr/bin/tar -xf "$PYTHON_ARCHIVE" -C "$STAGING/unpacked"
}

if [[ -x "$STAGING/unpacked/python/bin/python3" ]]; then
  RUNTIME_SOURCE="$STAGING/unpacked/python"
elif [[ -x "$STAGING/unpacked/bin/python3" ]]; then
  RUNTIME_SOURCE="$STAGING/unpacked"
else
  die "archive does not contain python/bin/python3"
fi

/usr/bin/ditto --noqtn "$RUNTIME_SOURCE" "$STAGING/runtime"
PYTHON="$STAGING/runtime/bin/python3"
[[ "$($PYTHON -c 'import platform; print(platform.machine())')" == "arm64" ]] || die "runtime is not arm64"
[[ "$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.12" ]] || die "runtime is not Python 3.12"

cd "$PROJECT_ROOT"
uv export --frozen --no-dev --no-emit-project --output-file "$STAGING/requirements.txt"
"$PYTHON" -m pip install \
  --disable-pip-version-check \
  --no-compile \
  --requirement "$STAGING/requirements.txt"

"$PYTHON" - <<'PY'
import fastapi
import mlx
import mlx_audio
import mlx_lm
import uvicorn
print("validated bundled dependencies")
PY

/usr/bin/find "$STAGING/runtime" \
  \( -name '__pycache__' -o -name '.pytest_cache' \) \
  -type d -prune -exec /bin/rm -rf {} +
/usr/bin/find "$STAGING/runtime" \
  \( -name '*.pyc' -o -name '*.pyo' -o -name '.DS_Store' \) \
  -type f -delete

/bin/rm -rf "$OUTPUT_DIR"
/bin/mv "$STAGING/runtime" "$OUTPUT_DIR"
echo "$OUTPUT_DIR"
