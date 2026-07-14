#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP=""
RELEASE_MODE=0
MAXIMUM_MACOS="14.0"

usage() {
  cat >&2 <<'EOF'
Usage: packaging/macos/verify-app.sh [--release] 'Noice City.app'

Without --release, signed and ad-hoc/local app bundles are accepted. When a
signature is present, it is still verified strictly. --release additionally
requires the pinned public-release metadata, icon, Developer ID signature,
hardened runtime, secure timestamp, and an empty entitlement set.
EOF
}

die() {
  echo "verify-app: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --release)
      [[ "$RELEASE_MODE" == "0" ]] || { usage; exit 64; }
      RELEASE_MODE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      [[ $# -eq 1 && -z "$APP" ]] || { usage; exit 64; }
      APP="$1"
      shift
      ;;
    -*)
      usage
      exit 64
      ;;
    *)
      [[ -z "$APP" ]] || { usage; exit 64; }
      APP="$1"
      shift
      ;;
  esac
done

if [[ ! -d "$APP/Contents" ]]; then
  usage
  exit 64
fi

# All subsequent checks change directories while validating checksum manifests.
# Canonicalize once so documented relative paths remain valid throughout.
APP="$(cd "$(dirname "$APP")" && pwd)/$(basename "$APP")"

require_file() {
  [[ -f "$1" ]] || die "required file is missing: ${1#$APP/}"
}

require_nonempty_file() {
  [[ -s "$1" ]] || die "required nonempty file is missing: ${1#$APP/}"
}

require_dir() {
  [[ -d "$1" ]] || die "required directory is missing: ${1#$APP/}"
}

# The runtime intentionally contains a small number of relative convenience
# links (for example, python -> python3.12). Never allow a packaged link to
# escape the signed bundle or depend on a release-machine absolute path.
while IFS= read -r -d '' link_path; do
  link_target="$(/usr/bin/readlink "$link_path")" ||
    die "could not read packaged symlink: ${link_path#$APP/}"
  [[ -n "$link_target" ]] ||
    die "packaged symlink has an empty target: ${link_path#$APP/}"
  [[ "$link_target" != /* ]] ||
    die "packaged symlink has an absolute target: ${link_path#$APP/} -> $link_target"
  case "/$link_target/" in
    */../*)
      die "packaged symlink traverses a parent directory: ${link_path#$APP/} -> $link_target"
      ;;
  esac
done < <(/usr/bin/find "$APP/Contents" -type l -print0)

verify_release_signature() {
  local candidate="$1"
  local signature_details
  local entitlements

  signature_details="$(/usr/bin/codesign -d --verbose=4 "$candidate" 2>&1)" ||
    die "could not read release signature: ${candidate#$APP/}"
  /usr/bin/grep -Fqx "Authority=$EXPECTED_AUTHORITY" <<<"$signature_details" ||
    die "release signature has the wrong Developer ID authority: ${candidate#$APP/}"
  /usr/bin/grep -Fqx "TeamIdentifier=$EXPECTED_TEAM_ID" <<<"$signature_details" ||
    die "release signature has the wrong team identifier: ${candidate#$APP/}"
  /usr/bin/grep -Eq '^CodeDirectory .*flags=.*\(runtime\)' <<<"$signature_details" ||
    die "release signature is missing the hardened runtime: ${candidate#$APP/}"
  if ! /usr/bin/grep -Eq '^Timestamp=.+$' <<<"$signature_details" ||
      /usr/bin/grep -Fqx 'Timestamp=none' <<<"$signature_details"; then
    die "release signature is missing a secure timestamp: ${candidate#$APP/}"
  fi

  entitlements="$(/usr/bin/codesign -d --entitlements - "$candidate" 2>/dev/null)" ||
    die "could not inspect release entitlements: ${candidate#$APP/}"
  if [[ -n "$entitlements" ]]; then
    if /usr/bin/grep -Fq 'com.apple.security.cs.allow-jit' <<<"$entitlements"; then
      die "release signature unexpectedly grants allow-jit: ${candidate#$APP/}"
    fi
    die "release signature contains unexpected entitlements: ${candidate#$APP/}"
  fi
}

CONTENTS="$APP/Contents"
RESOURCES="$CONTENTS/Resources"
PLIST="$CONTENTS/Info.plist"
RUNTIME_ROOT="$RESOURCES/runtime"
RUNTIME="$RUNTIME_ROOT/bin/python3"
APP_ROOT="$RESOURCES/app"
PYTHON_PROVENANCE="$RESOURCES/RELEASE_PROVENANCE/python-build-standalone"
PYTHON_LICENSES="$RESOURCES/THIRD_PARTY_LICENSES/python-build-standalone"
DEPENDENCY_PROVENANCE="$RESOURCES/RELEASE_PROVENANCE/runtime-dependencies"
SOURCE_PROVENANCE="$RESOURCES/RELEASE_PROVENANCE/SOURCE_COMMIT.txt"
RUNTIME_PACKAGE_LICENSES="$RESOURCES/THIRD_PARTY_LICENSES/runtime-packages"

/usr/bin/plutil -lint "$PLIST"
[[ -x "$CONTENTS/MacOS/NoiceCity" ]] || die "native launcher is missing or not executable"
[[ "$(/usr/bin/file -b "$CONTENTS/MacOS/NoiceCity")" == *Mach-O* ]] || die "native launcher is not Mach-O"
[[ -x "$RESOURCES/NoiceCity.launcher" ]] || die "server launcher is missing or not executable"
[[ -x "$RUNTIME" ]] || die "bundled python3 is missing or not executable"
require_dir "$APP_ROOT/voice_trainer"
require_dir "$APP_ROOT/static"
require_dir "$APP_ROOT/assets"
require_file "$APP_ROOT/assets/prompts.json"

if [[ "$RELEASE_MODE" == "1" ]]; then
  EXPECTED_PRODUCT_NAME="Noice City"
  EXPECTED_BUNDLE_ID="com.ruhitaj.noicecity"
  EXPECTED_VERSION="0.1.1"
  EXPECTED_BUILD="4"
  EXPECTED_MINIMUM_OS="$MAXIMUM_MACOS"
  EXPECTED_TEAM_ID="28DPUBPSZD"
  EXPECTED_AUTHORITY="Developer ID Application: Ruhitaj Reddypalli (28DPUBPSZD)"
  CANONICAL_ICON="$PROJECT_ROOT/assets/branding/NoiceCity.icns"

  plist_value() {
    /usr/libexec/PlistBuddy -c "Print :$1" "$PLIST" 2>/dev/null ||
      die "release Info.plist is missing $1"
  }

  require_plist_value() {
    local key="$1"
    local expected="$2"
    local actual
    actual="$(plist_value "$key")"
    [[ "$actual" == "$expected" ]] ||
      die "release Info.plist $key is '$actual', expected '$expected'"
  }

  require_plist_value CFBundleDisplayName "$EXPECTED_PRODUCT_NAME"
  require_plist_value CFBundleName "$EXPECTED_PRODUCT_NAME"
  require_plist_value CFBundleIdentifier "$EXPECTED_BUNDLE_ID"
  require_plist_value CFBundleShortVersionString "$EXPECTED_VERSION"
  require_plist_value CFBundleVersion "$EXPECTED_BUILD"
  require_plist_value LSMinimumSystemVersion "$EXPECTED_MINIMUM_OS"
  require_plist_value CFBundleIconFile AppIcon

  require_nonempty_file "$CANONICAL_ICON"
  require_nonempty_file "$RESOURCES/AppIcon.icns"
  /usr/bin/cmp -s "$CANONICAL_ICON" "$RESOURCES/AppIcon.icns" ||
    die "release icon differs from assets/branding/NoiceCity.icns"

  LAUNCHER_MINIMUM_OS="$(
    /usr/bin/vtool -show-build "$CONTENTS/MacOS/NoiceCity" 2>/dev/null |
      /usr/bin/awk '$1 == "minos" { print $2; exit }'
  )"
  [[ "$LAUNCHER_MINIMUM_OS" == "$EXPECTED_MINIMUM_OS" ]] ||
    die "native launcher minimum OS is '$LAUNCHER_MINIMUM_OS', expected '$EXPECTED_MINIMUM_OS'"
  LAUNCHER_ARCHS="$(/usr/bin/lipo -archs "$CONTENTS/MacOS/NoiceCity" 2>/dev/null)"
  [[ "$LAUNCHER_ARCHS" == "arm64" ]] ||
    die "native launcher architectures are '$LAUNCHER_ARCHS', expected 'arm64'"
fi

# Verify the complete code-signature envelope before executing anything from
# the bundle. Runtime import checks below are meaningful only after this trust
# boundary has passed.
APP_IS_SIGNED=0
if /usr/bin/codesign -d "$APP" >/dev/null 2>&1; then
  APP_IS_SIGNED=1
  while IFS= read -r -d '' candidate; do
    if /usr/bin/file -b "$candidate" | /usr/bin/grep -q 'Mach-O'; then
      /usr/bin/codesign --verify --strict --verbose=2 "$candidate"
      if [[ "$RELEASE_MODE" == "1" ]]; then
        verify_release_signature "$candidate"
      fi
    fi
  done < <(/usr/bin/find "$CONTENTS" -type f -print0)
  /usr/bin/codesign --verify --strict --verbose=2 "$APP"
  /usr/bin/codesign --verify --deep --strict --verbose=2 "$APP"
  if [[ "$RELEASE_MODE" == "1" ]]; then
    verify_release_signature "$APP"
  fi
fi

if [[ "$APP_IS_SIGNED" != "1" ]]; then
  die "app is unsigned; local verification requires at least an ad-hoc signature"
fi

for notice_file in LICENSE NOTICE TRADEMARKS.md THIRD_PARTY_NOTICES.md MODEL_NOTICES.md; do
  require_nonempty_file "$RESOURCES/$notice_file"
done
require_nonempty_file "$SOURCE_PROVENANCE"
for source_line in \
  'schema_version=1' \
  'repository=https://github.com/RuhiTaj/noice-city' \
  'worktree_clean=true'; do
  /usr/bin/grep -Fqx "$source_line" "$SOURCE_PROVENANCE" ||
    die "source provenance is missing: $source_line"
done
SOURCE_COMMIT="$(/usr/bin/awk -F= '$1 == "commit" { print $2 }' "$SOURCE_PROVENANCE")"
SOURCE_TREE="$(/usr/bin/awk -F= '$1 == "tree" { print $2 }' "$SOURCE_PROVENANCE")"
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ && "$SOURCE_TREE" =~ ^[0-9a-f]{40}$ ]] ||
  die "source provenance has invalid Git object identifiers"
for provenance_file in \
  PYTHON.json ARCHIVE_SHA256S.txt LICENSES_SHA256.txt SUPPLEMENTAL_LICENSE_PROVENANCE.txt; do
  require_nonempty_file "$PYTHON_PROVENANCE/$provenance_file"
done
require_dir "$PYTHON_LICENSES"
if ! /usr/bin/find "$PYTHON_LICENSES" -type f -print -quit | /usr/bin/grep -q .; then
  die "python-build-standalone component licenses are empty"
fi
for provenance_file in \
  uv.lock requirements.txt INSTALLED_PACKAGES.txt PIP_CHECK.txt \
  MLX_AUDIO_METADATA_OVERLAY.json MLX_TARGET_WHEELS.txt MACHO_COMPATIBILITY.txt \
  RUNTIME_PACKAGE_LICENSES_SHA256.txt \
  RUNTIME_PACKAGE_LICENSE_PROVENANCE.txt SHA256SUMS.txt; do
  require_nonempty_file "$DEPENDENCY_PROVENANCE/$provenance_file"
done
EXPECTED_MACHO_REPORT_HEADER=$'path\tarchitectures\tslice\tplatform\tminimum_macos\tsdk'
[[ "$(/usr/bin/head -n 1 "$DEPENDENCY_PROVENANCE/MACHO_COMPATIBILITY.txt")" == \
    "$EXPECTED_MACHO_REPORT_HEADER" ]] ||
  die "retained Mach-O compatibility report has an unexpected schema"
require_dir "$RUNTIME_PACKAGE_LICENSES"
if ! /usr/bin/find "$RUNTIME_PACKAGE_LICENSES" -type f -print -quit | /usr/bin/grep -q .; then
  die "runtime package licenses are empty"
fi
if ! /usr/bin/grep -Fqx \
  '3724aa4dafb5f7b6c2cf98e89914e4248dc6bd2fe40407df4a2d73de99615f16  cpython-3.12.13+20260623-aarch64-apple-darwin-install_only.tar.gz' \
  "$PYTHON_PROVENANCE/ARCHIVE_SHA256S.txt"; then
  die "install-only Python archive provenance is not the pinned release"
fi
if ! /usr/bin/grep -Fqx \
  'ce5a2d552077d869f69dc25d834c2fe4d036f9d78e770fb5d916273db802cabc  cpython-3.12.13+20260623-aarch64-apple-darwin-pgo+lto-full.tar.zst' \
  "$PYTHON_PROVENANCE/ARCHIVE_SHA256S.txt"; then
  die "full Python archive provenance is not the pinned release"
fi
if ! /usr/bin/grep -Fqx \
  'license SHA-256: 6c9f0d975b41afaa34d22f55bb8986ce69e5cb7ad327cb2b28820cd425edf5ee' \
  "$PYTHON_PROVENANCE/SUPPLEMENTAL_LICENSE_PROVENANCE.txt"; then
  die "supplemental zlib-ng license provenance is not pinned"
fi
(
  cd "$PYTHON_LICENSES"
  /usr/bin/shasum -a 256 -c "$PYTHON_PROVENANCE/LICENSES_SHA256.txt"
) || die "a Python component license does not match its release manifest"
(
  cd "$DEPENDENCY_PROVENANCE"
  /usr/bin/shasum -a 256 -c SHA256SUMS.txt
) || die "runtime dependency provenance does not match its release manifest"
(
  cd "$RUNTIME_PACKAGE_LICENSES"
  /usr/bin/shasum -a 256 -c "$DEPENDENCY_PROVENANCE/RUNTIME_PACKAGE_LICENSES_SHA256.txt"
) || die "a promoted runtime package license does not match its release manifest"

for exact_license in \
  'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4  tokenizers-0.22.2/LICENSE' \
  'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30  sentencepiece-0.2.2/LICENSE' \
  'fcff87c3a47ce8028a8512aa182d4fcf0ad1c90544ee75cf9b343684cac194de  tqdm-4.68.4/LICENCE'; do
  if ! /usr/bin/grep -Fqx "$exact_license" \
    "$DEPENDENCY_PROVENANCE/RUNTIME_PACKAGE_LICENSES_SHA256.txt"; then
    die "runtime package license manifest is not the reviewed release"
  fi
done
for pinned_dependency_evidence in \
  '1c5aae197285ecfda39cc3048f19f3c56ab22c8ac2dfa21d79ca6c7f37418ca7' \
  'ea5a594355c89c0095eaba413fd39d4caa8642fa13432dfb0c9354d141046467' \
  '5b64b20ac24b0c401f489de01e8209edc4d372125201f19314e6f39e385322aa' \
  '473b83b915e547aa366d1eee11806deaf419e17be16310ac0a14077f1e28f917' \
  '3d2b5e824b5622038dc7b490897efe05ebbbb9e7350fc142f3ecc8789ef9bdf6' \
  '19829c9673638f2a0b8617da4cdcb927e831cd88bcfcb6e78d42a4d1af131520'; do
  if ! /usr/bin/grep -R -Fq "$pinned_dependency_evidence" "$DEPENDENCY_PROVENANCE"; then
    die "pinned dependency evidence is missing: $pinned_dependency_evidence"
  fi
done
if ! /usr/bin/grep -Fq -- '--hash=sha256:' "$DEPENDENCY_PROVENANCE/requirements.txt"; then
  die "retained runtime requirements are not hash-pinned"
fi
if ! /usr/bin/grep -Fqx 'No broken requirements found.' "$DEPENDENCY_PROVENANCE/PIP_CHECK.txt"; then
  die "retained pip check result is not successful"
fi
for target_wheel_line in \
  'minimum_macos=14.0' \
  'required_architecture=arm64' \
  'target_platform=macosx_14_0_arm64' \
  'mlx_sha256=ea5a594355c89c0095eaba413fd39d4caa8642fa13432dfb0c9354d141046467' \
  'mlx_metal_sha256=5b64b20ac24b0c401f489de01e8209edc4d372125201f19314e6f39e385322aa'; do
  if ! /usr/bin/grep -Fqx "$target_wheel_line" "$DEPENDENCY_PROVENANCE/MLX_TARGET_WHEELS.txt"; then
    die "macOS 14 MLX target provenance is incomplete: $target_wheel_line"
  fi
done
for installed_package in \
  'fastapi==0.139.0' 'python-multipart==0.0.32' 'starlette==1.3.1' \
  'mlx-audio==0.4.5' 'mlx-lm==0.31.3' 'mlx==0.32.0' \
  'sentencepiece==0.2.2' 'tokenizers==0.22.2' 'tqdm==4.68.4'; do
  if ! /usr/bin/grep -Fqx "$installed_package" "$DEPENDENCY_PROVENANCE/INSTALLED_PACKAGES.txt"; then
    die "final package inventory is missing $installed_package"
  fi
done
if /usr/bin/grep -Eq '^(scipy|sounddevice)==' "$DEPENDENCY_PROVENANCE/INSTALLED_PACKAGES.txt"; then
  die "final package inventory contains a pruned dependency"
fi

# The application only needs the prompt catalog from repository-level assets.
# Finder artwork is copied separately as Resources/AppIcon.icns and web artwork
# lives under static/branding.
if /usr/bin/find "$APP_ROOT/assets" -mindepth 1 \
  ! -path "$APP_ROOT/assets/prompts.json" -print -quit | /usr/bin/grep -q .; then
  die "unallowlisted repository asset is present in the app bundle"
fi

RUNTIME_MACHINE="$("$RUNTIME" -I -B -c 'import platform; print(platform.machine())')"
RUNTIME_VERSION="$("$RUNTIME" -I -B -c 'import sys; print(sys.version.split()[0])')"
[[ "$RUNTIME_MACHINE" == "arm64" ]] || die "bundled runtime is $RUNTIME_MACHINE, expected arm64"
[[ "$RUNTIME_VERSION" == "3.12.13" ]] || die "bundled runtime is Python $RUNTIME_VERSION, expected 3.12.13"

# Generated console entry points contain preparation-machine shebangs and are
# unnecessary: the launcher executes python3 -m uvicorn.
if /usr/bin/find "$RUNTIME_ROOT/bin" -mindepth 1 -maxdepth 1 \
  ! -name python ! -name python3 ! -name python3.12 -print -quit | /usr/bin/grep -q .; then
  die "runtime/bin contains a non-interpreter entry point"
fi

# Mutable/private application data must never enter an allowlisted source tree.
for forbidden in \
  .venv state.sqlite session-token manifest.json recordings datasets checkpoints models \
  huggingface token keychain; do
  if /usr/bin/find "$APP_ROOT" -iname "$forbidden" -print -quit | /usr/bin/grep -q .; then
    die "forbidden private artifact in app source: $forbidden"
  fi
done

# Model weights, user audio, and local databases are never distributable app
# resources. This scan covers the interpreter as well as application source.
if /usr/bin/find "$CONTENTS" -type f \
  \( -iname '*.safetensors' -o -iname '*.ckpt' -o -iname '*.pth' -o -iname '*.pt' \
     -o -iname '*.onnx' -o -iname '*.gguf' -o -iname '*.npy' -o -iname '*.npz' \
     -o -iname '*.wav' -o -iname '*.flac' -o -iname '*.m4a' -o -iname '*.mp3' \
     -o -iname '*.aiff' -o -iname '*.aif' -o -iname '*.aac' -o -iname '*.ogg' \
     -o -iname '*.opus' -o -iname '*.sqlite' -o -iname '*.sqlite3' -o -iname '*.db' \
     -o -iname '*.sqlite-wal' -o -iname '*.sqlite-shm' \) \
  -print -quit | /usr/bin/grep -q .; then
  die "model, audio, or mutable database payload is present in the app bundle"
fi

# Certificate and private-key material is never application source. CA roots
# legitimately bundled by the Python runtime are outside APP_ROOT and are not
# matched by this source-only check.
if /usr/bin/find "$APP_ROOT" -type f \
  \( -iname '*.pem' -o -iname '*.key' -o -iname '*.p12' -o -iname '*.pfx' \
     -o -iname '*.cer' -o -iname '*.crt' -o -iname '*.mobileprovision' \) \
  -print -quit | /usr/bin/grep -q .; then
  die "credential or signing material is present in the app source"
fi

if /usr/bin/find "$CONTENTS" -type d \
  \( -name '__pycache__' -o -name '.pytest_cache' -o -name test -o -name tests \) \
  -print -quit | /usr/bin/grep -q .; then
  die "test or cache directory is present in the app bundle"
fi
if /usr/bin/find "$CONTENTS" -type f \
  \( -name '*.pyc' -o -name '*.pyo' -o -name '.DS_Store' \) \
  -print -quit | /usr/bin/grep -q .; then
  die "bytecode or filesystem cache is present in the app bundle"
fi

SITE_PACKAGES="$("$RUNTIME" -I -B -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for pruned_path in \
  "$SITE_PACKAGES/scipy" \
  "$SITE_PACKAGES/sounddevice.py" \
  "$SITE_PACKAGES/_sounddevice.py" \
  "$SITE_PACKAGES/_sounddevice_data" \
  "$SITE_PACKAGES/soundfile.py" \
  "$SITE_PACKAGES/_soundfile.py" \
  "$SITE_PACKAGES/_soundfile_data" \
  "$RUNTIME_ROOT/lib/python3.12/tkinter" \
  "$RUNTIME_ROOT/lib/python3.12/idlelib"; do
  [[ ! -e "$pruned_path" ]] || die "pruned runtime component is present: ${pruned_path#$APP/}"
done
if /usr/bin/find "$SITE_PACKAGES" -maxdepth 1 \
  \( -name 'scipy-*.dist-info' -o -name 'sounddevice-*.dist-info' \
     -o -name 'soundfile-*.dist-info' \) \
  -print -quit | /usr/bin/grep -q .; then
  die "pruned scipy, sounddevice, or soundfile package metadata is present"
fi
if /usr/bin/find "$RUNTIME_ROOT" \
  \( -iname '*libsndfile*' -o -iname '*portaudio*' -o -iname '*libgfortran*' \
     -o -iname '*libquadmath*' -o -iname 'libgcc_s*' \) \
  -print -quit | /usr/bin/grep -q .; then
  die "pruned native audio or compiler runtime is present"
fi
if /usr/bin/find "$RUNTIME_ROOT/lib/python3.12/lib-dynload" -maxdepth 1 \
  -name '_tkinter*.so' -print -quit | /usr/bin/grep -q .; then
  die "pruned _tkinter extension is present"
fi
if /usr/bin/find "$RUNTIME_ROOT/lib" -maxdepth 1 \
  \( -name 'tcl*' -o -name 'tk*' -o -name 'itcl*' -o -name 'thread*' \
     -o -name 'libtcl*' -o -name 'libtk*' -o -name 'libitcl*' -o -name 'libthread*' \) \
  -print -quit | /usr/bin/grep -q .; then
  die "pruned Tcl/Tk runtime component is present"
fi

# Scan text and binary payloads for this release machine's home/project paths.
# Console entry points were the primary leak, but this catches regressions
# elsewhere without embedding a developer-specific username in public source.
LEAKED_VALUES=("$PROJECT_ROOT" '.runtime-staging.' '.app-staging.')
if [[ "${HOME:-}" == /Users/* ]]; then
  LEAKED_VALUES+=("$HOME")
fi
for leaked_value in "${LEAKED_VALUES[@]}"; do
  if /usr/bin/grep -a -R -l -F -- "$leaked_value" "$CONTENTS" 2>/dev/null | /usr/bin/grep -q .; then
    die "release-machine path leaked into app bundle: $leaked_value"
  fi
done

NOICE_VERIFY_RESOURCES="$RESOURCES" "$RUNTIME" -I -B - <<'PY'
import base64
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path

import fastapi
import mlx
import mlx_audio
import mlx_lm
import uvicorn
from mlx_audio.stt.utils import load as load_stt
from mlx_audio.tts.utils import load_model as load_tts

resources = Path(os.environ["NOICE_VERIFY_RESOURCES"])
dependency_provenance = resources / "RELEASE_PROVENANCE/runtime-dependencies"
metadata = json.loads(
    (resources / "RELEASE_PROVENANCE/python-build-standalone/PYTHON.json").read_text(
        encoding="utf-8"
    )
)
expected_metadata = {
    "python_version": "3.12.13",
    "target_triple": "aarch64-apple-darwin",
    "optimizations": "pgo+lto",
}
for key, expected in expected_metadata.items():
    if metadata.get(key) != expected:
        raise SystemExit(f"unexpected Python provenance {key}: {metadata.get(key)!r}")

overlay = json.loads(
    (dependency_provenance / "MLX_AUDIO_METADATA_OVERLAY.json").read_text(encoding="utf-8")
)
expected_overlay = {
    "schema_version": 1,
    "package": "mlx-audio",
    "version": "0.4.5",
    "removed_requires_dist": ["scipy>=1.10.0", "sounddevice>=0.5.3"],
    "record_entry_updated": "mlx_audio-0.4.5.dist-info/METADATA",
    "package_code_modified": False,
}
for key, expected in expected_overlay.items():
    if overlay.get(key) != expected:
        raise SystemExit(f"unexpected MLX-Audio overlay {key}: {overlay.get(key)!r}")
if overlay.get("upstream_wheel") != {
    "url": "https://files.pythonhosted.org/packages/ca/59/af94af1cd78242893826826996c4407a6f50c766d41afd424d332d294975/mlx_audio-0.4.5-py3-none-any.whl",
    "sha256": "1c5aae197285ecfda39cc3048f19f3c56ab22c8ac2dfa21d79ca6c7f37418ca7",
}:
    raise SystemExit("MLX-Audio overlay does not identify the pinned upstream wheel")
if overlay.get("original_metadata") != {
    "sha256": "734fb0cdc3c11446b825f2ec14a09a7abf50b1c9f079b688f979aafc84cea596",
    "size": 31501,
}:
    raise SystemExit("MLX-Audio original metadata provenance changed")
if overlay.get("modified_metadata") != {
    "sha256": "1ae6206bc3022f2abb5b4314cd0ea4add510a923eadaf46bc567b85439baf235",
    "size": 31438,
}:
    raise SystemExit("MLX-Audio modified metadata provenance changed")

mlx_audio_dist = importlib.metadata.distribution("mlx-audio")
dist_info = Path(mlx_audio_dist._path)
metadata_path = dist_info / "METADATA"
record_path = dist_info / "RECORD"
metadata_bytes = metadata_path.read_bytes()
record_bytes = record_path.read_bytes()
if hashlib.sha256(metadata_bytes).hexdigest() != overlay["modified_metadata"]["sha256"]:
    raise SystemExit("installed MLX-Audio METADATA does not match the overlay")
if len(metadata_bytes) != overlay["modified_metadata"]["size"]:
    raise SystemExit("installed MLX-Audio METADATA size does not match the overlay")
if hashlib.sha256(record_bytes).hexdigest() != overlay["modified_record"]["sha256"]:
    raise SystemExit("installed MLX-Audio RECORD does not match the overlay")
if len(record_bytes) != overlay["modified_record"]["size"]:
    raise SystemExit("installed MLX-Audio RECORD size does not match the overlay")
for recorded in (overlay.get("original_record"), overlay.get("modified_record")):
    if not isinstance(recorded, dict) or len(recorded.get("sha256", "")) != 64:
        raise SystemExit("MLX-Audio RECORD provenance is incomplete")

encoded_hash = base64.urlsafe_b64encode(hashlib.sha256(metadata_bytes).digest()).rstrip(b"=")
expected_record = (
    b"mlx_audio-0.4.5.dist-info/METADATA,sha256="
    + encoded_hash
    + b"," + str(len(metadata_bytes)).encode("ascii")
)
record_matches = [line for line in record_bytes.splitlines() if line.startswith(
    b"mlx_audio-0.4.5.dist-info/METADATA,"
)]
if record_matches != [expected_record]:
    raise SystemExit("MLX-Audio RECORD does not contain the updated METADATA hash")
requires = mlx_audio_dist.requires or []
for removed_requirement in ("scipy", "sounddevice"):
    if any(requirement.lower().startswith(removed_requirement) for requirement in requires):
        raise SystemExit(f"MLX-Audio still declares pruned dependency: {removed_requirement}")

installed: dict[str, str] = {}
for dist in importlib.metadata.distributions():
    name = dist.metadata.get("Name")
    if not name:
        raise SystemExit(f"installed distribution has no Name: {dist._path}")
    canonical = name.lower().replace("_", "-")
    if canonical in installed:
        raise SystemExit(f"duplicate installed distribution: {canonical}")
    installed[canonical] = dist.version
current_inventory = "".join(f"{name}=={installed[name]}\n" for name in sorted(installed))
retained_inventory = (dependency_provenance / "INSTALLED_PACKAGES.txt").read_text(
    encoding="utf-8"
)
if current_inventory != retained_inventory:
    raise SystemExit("current installed package inventory differs from release provenance")
expected_wheel_tags = {
    "mlx": "cp312-cp312-macosx_14_0_arm64",
    "mlx-metal": "py3-none-macosx_14_0_arm64",
}
for distribution_name, expected_tag in expected_wheel_tags.items():
    distribution = importlib.metadata.distribution(distribution_name)
    wheel_path = Path(distribution._path) / "WHEEL"
    if not wheel_path.is_file():
        raise SystemExit(f"{distribution_name} WHEEL metadata is missing")
    tags = [
        line.removeprefix("Tag: ")
        for line in wheel_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("Tag: ")
    ]
    if tags != [expected_tag]:
        raise SystemExit(
            f"{distribution_name} has wheel tags {tags!r}, expected {[expected_tag]!r}"
        )
for pruned in ("scipy", "sounddevice", "tkinter", "idlelib"):
    if importlib.util.find_spec(pruned) is not None:
        raise SystemExit(f"pruned package remains importable: {pruned}")
print("validated bundled imports")
PY

PIP_CHECK_RESULT="$("$RUNTIME" -I -B -m pip check)"
[[ "$PIP_CHECK_RESULT" == 'No broken requirements found.' ]] || \
  die "bundled runtime fails pip check: $PIP_CHECK_RESULT"

"$SCRIPT_DIR/audit-macho.sh" \
  --root "$CONTENTS" \
  --maximum-macos "$MAXIMUM_MACOS" \
  --required-arch arm64

echo "Verified: $APP"
