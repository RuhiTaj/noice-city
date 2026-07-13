#!/usr/bin/env bash
set -euo pipefail

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# These are the exact companion artifacts used for the 3.12.13 arm64 release
# runtime. The install-only archive is what ships. The full archive is required
# at release time because it contains PYTHON.json and the component licenses
# omitted by install-only archives.
PYTHON_INSTALL_SHA256="3724aa4dafb5f7b6c2cf98e89914e4248dc6bd2fe40407df4a2d73de99615f16"
PYTHON_FULL_SHA256="ce5a2d552077d869f69dc25d834c2fe4d036f9d78e770fb5d916273db802cabc"
PYTHON_INSTALL_CANONICAL_NAME="cpython-3.12.13+20260623-aarch64-apple-darwin-install_only.tar.gz"
PYTHON_FULL_CANONICAL_NAME="cpython-3.12.13+20260623-aarch64-apple-darwin-pgo+lto-full.tar.zst"
ZLIB_NG_LICENSE_SHA256="6c9f0d975b41afaa34d22f55bb8986ce69e5cb7ad327cb2b28820cd425edf5ee"
ZLIB_NG_SOURCE_SHA256="00bbd88709bc416cb96160ab61d3e1c8f76e106799af7328d0fe434dc7dd5004"
ZLIB_NG_LICENSE="$SCRIPT_DIR/licenses/LICENSE.zlib-ng.txt"
MLX_AUDIO_WHEEL_URL="https://files.pythonhosted.org/packages/ca/59/af94af1cd78242893826826996c4407a6f50c766d41afd424d332d294975/mlx_audio-0.4.5-py3-none-any.whl"
MLX_AUDIO_WHEEL_SHA256="1c5aae197285ecfda39cc3048f19f3c56ab22c8ac2dfa21d79ca6c7f37418ca7"
MLX_AUDIO_ORIGINAL_METADATA_SHA256="734fb0cdc3c11446b825f2ec14a09a7abf50b1c9f079b688f979aafc84cea596"
MLX_AUDIO_MODIFIED_METADATA_SHA256="1ae6206bc3022f2abb5b4314cd0ea4add510a923eadaf46bc567b85439baf235"
TOKENIZERS_LICENSE_SHA256="c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
SENTENCEPIECE_LICENSE_SHA256="cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
TQDM_LICENSE_SHA256="fcff87c3a47ce8028a8512aa182d4fcf0ad1c90544ee75cf9b343684cac194de"
TARGET_MACOS_VERSION="14.0"
MLX_TARGET_PLATFORM="macosx_14_0_arm64"
MLX_TARGET_WHEEL_URL="https://files.pythonhosted.org/packages/aa/a6/489176d8a2a06137a299910057cc44dc3fccfb73151f7562fea2b75894d9/mlx-0.32.0-cp312-cp312-macosx_14_0_arm64.whl"
MLX_TARGET_WHEEL_SHA256="ea5a594355c89c0095eaba413fd39d4caa8642fa13432dfb0c9354d141046467"
MLX_METAL_TARGET_WHEEL_URL="https://files.pythonhosted.org/packages/d6/ef/d74ae99cfe9ddb59fd08abd14f47754c3d199291a93756903fa595b31b8a/mlx_metal-0.32.0-py3-none-macosx_14_0_arm64.whl"
MLX_METAL_TARGET_WHEEL_SHA256="5b64b20ac24b0c401f489de01e8209edc4d372125201f19314e6f39e385322aa"

PYTHON_ARCHIVE=""
PYTHON_FULL_ARCHIVE=""
OUTPUT_DIR="$PROJECT_ROOT/build/macos/runtime"

usage() {
  cat <<'EOF'
Prepare a relocatable Apple Silicon Python runtime for the .app bundle.

Usage:
  packaging/macos/prepare-runtime.sh \
    --python-archive FILE-install_only.tar.gz \
    --python-full-archive FILE-full.tar.zst \
    [--output DIR]

Both archives are required and must match the hard-coded SHA-256 values for
the pinned python-build-standalone CPython 3.12.13 arm64 release. The full
archive supplies PYTHON.json and component licenses omitted from install-only
archives. The script verifies that every shipped install-only file matches its
counterpart under python/install in the full archive.

This is a release-build step and may resolve the locked Python dependencies.
It never downloads application models or copies the project's .venv.
EOF
}

die() {
  echo "prepare-runtime: $*" >&2
  exit 64
}

sha256_file() {
  /usr/bin/shasum -a 256 "$1" | /usr/bin/awk '{print $1}'
}

verify_archive() {
  local archive="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(sha256_file "$archive")"
  [[ "$actual" == "$expected" ]] || die "$label SHA-256 mismatch: expected $expected, got $actual"
}

extract_archive() {
  local archive="$1"
  local destination="$2"
  mkdir -p "$destination"
  if /usr/bin/tar -xf "$archive" -C "$destination" 2>/dev/null; then
    return
  fi
  /bin/rm -rf "$destination"
  mkdir -p "$destination"
  if command -v zstd >/dev/null 2>&1; then
    zstd -q -d -c "$archive" | /usr/bin/tar -xf - -C "$destination"
    return
  fi
  die "could not extract $archive; install zstd if this macOS tar lacks .tar.zst support"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-archive)
      [[ $# -ge 2 && -n "$2" ]] || die "--python-archive requires a file"
      PYTHON_ARCHIVE="$2"
      shift 2
      ;;
    --python-full-archive)
      [[ $# -ge 2 && -n "$2" ]] || die "--python-full-archive requires a file"
      PYTHON_FULL_ARCHIVE="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 && -n "$2" ]] || die "--output requires a directory"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$(uname -s)" == "Darwin" ]] || die "must run on macOS"
[[ "$(uname -m)" == "arm64" ]] || die "must run on Apple Silicon"
[[ -n "$PYTHON_ARCHIVE" ]] || die "--python-archive is required"
[[ -n "$PYTHON_FULL_ARCHIVE" ]] || die "--python-full-archive is required"
[[ -f "$PYTHON_ARCHIVE" ]] || die "archive not found: $PYTHON_ARCHIVE"
[[ -f "$PYTHON_FULL_ARCHIVE" ]] || die "full archive not found: $PYTHON_FULL_ARCHIVE"
command -v uv >/dev/null 2>&1 || die "uv is required for a locked dependency export"

verify_archive "$PYTHON_ARCHIVE" "$PYTHON_INSTALL_SHA256" "install-only archive"
verify_archive "$PYTHON_FULL_ARCHIVE" "$PYTHON_FULL_SHA256" "full archive"
[[ -f "$ZLIB_NG_LICENSE" ]] || die "supplemental zlib-ng license is missing: $ZLIB_NG_LICENSE"
verify_archive "$ZLIB_NG_LICENSE" "$ZLIB_NG_LICENSE_SHA256" "supplemental zlib-ng license"

mkdir -p "$(dirname "$OUTPUT_DIR")"
OUTPUT_PARENT="$(cd "$(dirname "$OUTPUT_DIR")" && pwd)"
STAGING="$(mktemp -d "$OUTPUT_PARENT/.runtime-staging.XXXXXX")"
trap 'rm -rf "$STAGING"' EXIT

extract_archive "$PYTHON_ARCHIVE" "$STAGING/install-only"
extract_archive "$PYTHON_FULL_ARCHIVE" "$STAGING/full"

if [[ -x "$STAGING/install-only/python/bin/python3" ]]; then
  RUNTIME_SOURCE="$STAGING/install-only/python"
elif [[ -x "$STAGING/install-only/bin/python3" ]]; then
  RUNTIME_SOURCE="$STAGING/install-only"
else
  die "install-only archive does not contain python/bin/python3"
fi

FULL_PYTHON_ROOT="$STAGING/full/python"
[[ -f "$FULL_PYTHON_ROOT/PYTHON.json" ]] || die "full archive does not contain python/PYTHON.json"
[[ -x "$FULL_PYTHON_ROOT/install/bin/python3" ]] || die "full archive does not contain python/install/bin/python3"

/usr/bin/ditto --noqtn "$RUNTIME_SOURCE" "$STAGING/runtime"
PYTHON="$STAGING/runtime/bin/python3"
[[ "$($PYTHON -I -B -c 'import platform; print(platform.machine())')" == "arm64" ]] || die "runtime is not arm64"
[[ "$($PYTHON -I -B -c 'import sys; print(sys.version.split()[0])')" == "3.12.13" ]] || die "runtime is not Python 3.12.13"

RELEASE_ROOT="$STAGING/runtime/share/noice-city/release"
PROVENANCE_ROOT="$RELEASE_ROOT/python-build-standalone"
DEPENDENCY_PROVENANCE_ROOT="$RELEASE_ROOT/runtime-dependencies"
RUNTIME_PACKAGE_LICENSE_ROOT="$RELEASE_ROOT/runtime-package-licenses"
mkdir -p "$PROVENANCE_ROOT/licenses"
mkdir -p "$DEPENDENCY_PROVENANCE_ROOT" "$RUNTIME_PACKAGE_LICENSE_ROOT"

# Verify that the exact install-only tree being shipped came from the pinned
# full archive, then collect every component license referenced by PYTHON.json.
"$PYTHON" -I -B - \
  "$RUNTIME_SOURCE" \
  "$FULL_PYTHON_ROOT" \
  "$PROVENANCE_ROOT" \
  "$SCRIPT_DIR/licenses" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath

install_root = Path(sys.argv[1]).resolve()
full_root = Path(sys.argv[2]).resolve()
provenance_root = Path(sys.argv[3]).resolve()
supplement_root = Path(sys.argv[4]).resolve()
full_install = full_root / "install"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


mismatches: list[str] = []
checked = 0
for source in install_root.rglob("*"):
    relative = source.relative_to(install_root)
    counterpart = full_install / relative
    checked += 1
    if source.is_symlink():
        if not counterpart.is_symlink() or os.readlink(source) != os.readlink(counterpart):
            mismatches.append(f"symlink differs: {relative}")
    elif source.is_dir():
        if not counterpart.is_dir():
            mismatches.append(f"directory missing: {relative}")
    elif source.is_file():
        if not counterpart.is_file() or digest(source) != digest(counterpart):
            mismatches.append(f"file differs: {relative}")
    else:
        mismatches.append(f"unsupported entry: {relative}")

if checked < 1000:
    mismatches.append(f"install-only archive unexpectedly contains only {checked} entries")
if mismatches:
    preview = "\n".join(f"  - {item}" for item in mismatches[:25])
    raise SystemExit(f"install-only/full archive mismatch:\n{preview}")

metadata_path = full_root / "PYTHON.json"
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
if metadata.get("python_version") != "3.12.13":
    raise SystemExit(f"unexpected PYTHON.json python_version: {metadata.get('python_version')!r}")
if metadata.get("target_triple") != "aarch64-apple-darwin":
    raise SystemExit(f"unexpected PYTHON.json target_triple: {metadata.get('target_triple')!r}")
if metadata.get("optimizations") != "pgo+lto":
    raise SystemExit(f"unexpected PYTHON.json optimizations: {metadata.get('optimizations')!r}")

license_paths: set[str] = set()
if metadata.get("license_path"):
    license_paths.add(str(metadata["license_path"]))
for variants in metadata.get("build_info", {}).get("extensions", {}).values():
    for variant in variants:
        license_paths.update(str(path) for path in (variant.get("license_paths") or []))
if not license_paths:
    raise SystemExit("PYTHON.json does not reference any component licenses")

license_output = provenance_root / "licenses"
license_manifest: list[str] = []
for raw_path in sorted(license_paths):
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise SystemExit(f"unsafe license_path in PYTHON.json: {raw_path!r}")
    source = full_root.joinpath(*relative.parts)
    if not source.is_file():
        # The pinned 20260623 full archive references this exact zlib-ng
        # license_path but accidentally omits the file. Its exact upstream
        # LICENSE.md is vendored and hash-verified by the shell wrapper.
        if raw_path == "licenses/LICENSE.zlib-ng.txt":
            source = supplement_root / "LICENSE.zlib-ng.txt"
        if not source.is_file():
            raise SystemExit(f"PYTHON.json license_path is missing from full archive: {raw_path}")
    output_parts = relative.parts[1:] if relative.parts[0] == "licenses" else relative.parts
    if not output_parts:
        raise SystemExit(f"invalid license_path in PYTHON.json: {raw_path!r}")
    destination = license_output.joinpath(*output_parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    license_manifest.append(f"{digest(destination)}  {destination.relative_to(license_output).as_posix()}")

shutil.copy2(metadata_path, provenance_root / "PYTHON.json")
(provenance_root / "LICENSES_SHA256.txt").write_text(
    "\n".join(license_manifest) + "\n",
    encoding="utf-8",
)
print(f"validated {checked} install-only entries and collected {len(license_paths)} licenses")
PY

/usr/bin/printf '%s  %s\n%s  %s\n' \
  "$PYTHON_INSTALL_SHA256" "$PYTHON_INSTALL_CANONICAL_NAME" \
  "$PYTHON_FULL_SHA256" "$PYTHON_FULL_CANONICAL_NAME" \
  > "$PROVENANCE_ROOT/ARCHIVE_SHA256S.txt"
/usr/bin/printf '%s\n' \
  'The pinned full Python archive references licenses/LICENSE.zlib-ng.txt but omits it.' \
  'The bundled LICENSE.zlib-ng.txt is an exact copy of LICENSE.md from the pinned source archive:' \
  'https://github.com/python/cpython-source-deps/archive/refs/tags/zlib-ng-2.2.4.tar.gz' \
  "source archive SHA-256: $ZLIB_NG_SOURCE_SHA256" \
  "license SHA-256: $ZLIB_NG_LICENSE_SHA256" \
  > "$PROVENANCE_ROOT/SUPPLEMENTAL_LICENSE_PROVENANCE.txt"

cd "$PROJECT_ROOT"
[[ -s "$PROJECT_ROOT/uv.lock" ]] || die "uv.lock is missing"
[[ -s "$PROJECT_ROOT/LICENSE" ]] || die "repository Apache-2.0 license is missing"
verify_archive "$PROJECT_ROOT/LICENSE" "$TOKENIZERS_LICENSE_SHA256" "repository Apache-2.0 license"
uv export --frozen --no-dev --no-emit-project > "$STAGING/requirements.txt"
if ! /usr/bin/grep -Fq -- '--hash=sha256:' "$STAGING/requirements.txt"; then
  die "uv export did not produce hashed requirements"
fi
for reviewed_lock_hash in \
  "$MLX_AUDIO_WHEEL_SHA256" \
  "$MLX_TARGET_WHEEL_SHA256" \
  "$MLX_METAL_TARGET_WHEEL_SHA256" \
  473b83b915e547aa366d1eee11806deaf419e17be16310ac0a14077f1e28f917 \
  1e418a55456beedca4621dbab65a318981467a2b188e982a23e117f115ce5001 \
  3d2b5e824b5622038dc7b490897efe05ebbbb9e7350fc142f3ecc8789ef9bdf6 \
  38111ed1f79268f399c505028023d5eaaf0ab4e5eafceb709468b0d3323e7838 \
  19829c9673638f2a0b8617da4cdcb927e831cd88bcfcb6e78d42a4d1af131520 \
  5168118b2368f48c561afda8020fd79195b1bdb0bdf8086b88442c267a315dc2; do
  if ! /usr/bin/grep -Fq "$reviewed_lock_hash" "$PROJECT_ROOT/uv.lock"; then
    die "uv.lock is missing reviewed artifact hash: $reviewed_lock_hash"
  fi
done
if ! /usr/bin/grep -Fq "$MLX_AUDIO_WHEEL_SHA256" "$STAGING/requirements.txt"; then
  die "exported requirements omit the pinned MLX-Audio wheel hash"
fi
for target_wheel_hash in "$MLX_TARGET_WHEEL_SHA256" "$MLX_METAL_TARGET_WHEEL_SHA256"; do
  if ! /usr/bin/grep -Fq "$target_wheel_hash" "$STAGING/requirements.txt"; then
    die "exported requirements omit the pinned macOS 14 MLX wheel hash: $target_wheel_hash"
  fi
done
/bin/cp "$PROJECT_ROOT/uv.lock" "$DEPENDENCY_PROVENANCE_ROOT/uv.lock"
/bin/cp "$STAGING/requirements.txt" "$DEPENDENCY_PROVENANCE_ROOT/requirements.txt"

# Install the two platform-sensitive MLX distributions first from their exact
# lock-pinned macOS 14 wheels. A normal host-platform resolution on a newer
# build Mac would otherwise select that host's wheel and silently raise the
# finished app's deployment target. The complete hashed requirements install
# below sees these exact versions as already satisfied.
"$PYTHON" -I -m pip install \
  --disable-pip-version-check \
  --no-compile \
  --no-deps \
  --only-binary=:all: \
  --require-hashes \
  "mlx @ ${MLX_TARGET_WHEEL_URL}#sha256=${MLX_TARGET_WHEEL_SHA256}" \
  "mlx-metal @ ${MLX_METAL_TARGET_WHEEL_URL}#sha256=${MLX_METAL_TARGET_WHEEL_SHA256}"
"$PYTHON" -I -m pip install \
  --disable-pip-version-check \
  --no-compile \
  --only-binary=:all: \
  --require-hashes \
  --requirement "$STAGING/requirements.txt"

# VoiceTrainer decodes the PCM WAVs it owns and uses miniaudio for Whisper.
# It never calls MLX-Audio's optional scipy resampler or sounddevice players.
SITE_PACKAGES="$($PYTHON -I -B -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

/usr/bin/printf '%s\n' \
  "minimum_macos=$TARGET_MACOS_VERSION" \
  'required_architecture=arm64' \
  "target_platform=$MLX_TARGET_PLATFORM" \
  "mlx_url=$MLX_TARGET_WHEEL_URL" \
  "mlx_sha256=$MLX_TARGET_WHEEL_SHA256" \
  "mlx_metal_url=$MLX_METAL_TARGET_WHEEL_URL" \
  "mlx_metal_sha256=$MLX_METAL_TARGET_WHEEL_SHA256" \
  > "$DEPENDENCY_PROVENANCE_ROOT/MLX_TARGET_WHEELS.txt"

"$PYTHON" -I -B - "$SITE_PACKAGES" "$MLX_TARGET_PLATFORM" <<'PY'
from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

site_packages = Path(sys.argv[1]).resolve()
target_platform = sys.argv[2]
expected = {
    "mlx": f"cp312-cp312-{target_platform}",
    "mlx-metal": f"py3-none-{target_platform}",
}
for distribution_name, expected_tag in expected.items():
    distribution = importlib.metadata.distribution(distribution_name)
    dist_info = Path(distribution._path).resolve()
    if site_packages not in dist_info.parents:
        raise SystemExit(f"{distribution_name} was not installed in the bundled runtime")
    wheel = dist_info / "WHEEL"
    if not wheel.is_file():
        raise SystemExit(f"{distribution_name} WHEEL metadata is missing")
    tags = [
        line.removeprefix("Tag: ")
        for line in wheel.read_text(encoding="utf-8").splitlines()
        if line.startswith("Tag: ")
    ]
    if tags != [expected_tag]:
        raise SystemExit(
            f"{distribution_name} has wheel tags {tags!r}, expected {[expected_tag]!r}"
        )
print("validated exact macOS 14 MLX wheel tags")
PY

# tokenizers and sentencepiece publish Apache-2.0 license texts in their source
# distributions but omit them from the macOS wheels. The repository's own
# Apache-2.0 LICENSE is byte-for-byte tokenizers' text; sentencepiece's exact
# text differs only by one leading newline. tqdm does retain its exact LICENCE
# inside the wheel. Copy and hash all three into an obvious bundle location.
"$PYTHON" -I -B - \
  "$PROJECT_ROOT/LICENSE" \
  "$RUNTIME_PACKAGE_LICENSE_ROOT" \
  "$DEPENDENCY_PROVENANCE_ROOT" \
  "$SITE_PACKAGES" \
  "$TOKENIZERS_LICENSE_SHA256" \
  "$SENTENCEPIECE_LICENSE_SHA256" \
  "$TQDM_LICENSE_SHA256" <<'PY'
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

root_license = Path(sys.argv[1])
license_root = Path(sys.argv[2])
provenance_root = Path(sys.argv[3])
site_packages = Path(sys.argv[4])
tokenizers_hash, sentencepiece_hash, tqdm_hash = sys.argv[5:8]


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


apache = root_license.read_bytes()
if digest_bytes(apache) != tokenizers_hash:
    raise SystemExit("repository LICENSE no longer matches tokenizers 0.22.2")
sentencepiece = b"\n" + apache
if digest_bytes(sentencepiece) != sentencepiece_hash:
    raise SystemExit("constructed sentencepiece 0.2.2 license hash mismatch")

tokenizers_output = license_root / "tokenizers-0.22.2/LICENSE"
sentencepiece_output = license_root / "sentencepiece-0.2.2/LICENSE"
tqdm_source = site_packages / "tqdm-4.68.4.dist-info/licenses/LICENCE"
tqdm_output = license_root / "tqdm-4.68.4/LICENCE"
if not tqdm_source.is_file() or digest(tqdm_source) != tqdm_hash:
    raise SystemExit("tqdm 4.68.4 wheel license is missing or changed")

for output in (tokenizers_output, sentencepiece_output, tqdm_output):
    output.parent.mkdir(parents=True, exist_ok=True)
tokenizers_output.write_bytes(apache)
sentencepiece_output.write_bytes(sentencepiece)
shutil.copy2(tqdm_source, tqdm_output)

manifest = [
    f"{tokenizers_hash}  tokenizers-0.22.2/LICENSE",
    f"{sentencepiece_hash}  sentencepiece-0.2.2/LICENSE",
    f"{tqdm_hash}  tqdm-4.68.4/LICENCE",
]
(provenance_root / "RUNTIME_PACKAGE_LICENSES_SHA256.txt").write_text(
    "\n".join(manifest) + "\n",
    encoding="utf-8",
)
(provenance_root / "RUNTIME_PACKAGE_LICENSE_PROVENANCE.txt").write_text(
    "\n".join(
        [
            "tokenizers 0.22.2: exact Apache-2.0 LICENSE from the pinned source distribution",
            "source distribution SHA-256: 473b83b915e547aa366d1eee11806deaf419e17be16310ac0a14077f1e28f917",
            "macOS arm64 wheel SHA-256: 1e418a55456beedca4621dbab65a318981467a2b188e982a23e117f115ce5001",
            "sentencepiece 0.2.2: exact Apache-2.0 LICENSE from the pinned source distribution",
            "source distribution SHA-256: 3d2b5e824b5622038dc7b490897efe05ebbbb9e7350fc142f3ecc8789ef9bdf6",
            "macOS arm64 wheel SHA-256: 38111ed1f79268f399c505028023d5eaaf0ab4e5eafceb709468b0d3323e7838",
            "tqdm 4.68.4: exact wheel-provided dist-info/licenses/LICENCE",
            "source distribution SHA-256: 19829c9673638f2a0b8617da4cdcb927e831cd88bcfcb6e78d42a4d1af131520",
            "wheel SHA-256: 5168118b2368f48c561afda8020fd79195b1bdb0bdf8086b88442c267a315dc2",
        ]
    )
    + "\n",
    encoding="utf-8",
)
PY

/bin/rm -rf \
  "$SITE_PACKAGES/scipy" \
  "$SITE_PACKAGES"/scipy-*.dist-info \
  "$SITE_PACKAGES/sounddevice.py" \
  "$SITE_PACKAGES/_sounddevice.py" \
  "$SITE_PACKAGES/_sounddevice_data" \
  "$SITE_PACKAGES"/sounddevice-*.dist-info

# MLX-Audio 0.4.5 currently declares SciPy and SoundDevice as unconditional
# requirements even though Noice City does not exercise those optional paths.
# After pruning those packages, remove exactly those two Requires-Dist lines,
# update only METADATA's entry in RECORD, and retain a hash-complete audit of
# the original wheel metadata and the release overlay. No package code changes.
"$PYTHON" -I -B - \
  "$SITE_PACKAGES" \
  "$DEPENDENCY_PROVENANCE_ROOT" \
  "$MLX_AUDIO_WHEEL_URL" \
  "$MLX_AUDIO_WHEEL_SHA256" \
  "$MLX_AUDIO_ORIGINAL_METADATA_SHA256" \
  "$MLX_AUDIO_MODIFIED_METADATA_SHA256" <<'PY'
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from pathlib import Path

site_packages = Path(sys.argv[1])
provenance_root = Path(sys.argv[2])
wheel_url, wheel_hash, expected_original_hash, expected_modified_hash = sys.argv[3:7]
dist_info = site_packages / "mlx_audio-0.4.5.dist-info"
metadata_path = dist_info / "METADATA"
record_path = dist_info / "RECORD"
if not metadata_path.is_file() or not record_path.is_file():
    raise SystemExit("mlx-audio 0.4.5 dist-info is missing")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


original_metadata = metadata_path.read_bytes()
original_record = record_path.read_bytes()
if digest(original_metadata) != expected_original_hash:
    raise SystemExit("mlx-audio 0.4.5 original METADATA hash does not match the pinned wheel")

removed = [
    b"Requires-Dist: scipy>=1.10.0\n",
    b"Requires-Dist: sounddevice>=0.5.3\n",
]
modified_metadata = original_metadata
for line in removed:
    if modified_metadata.count(line) != 1:
        raise SystemExit(f"expected exactly one metadata line: {line.decode().rstrip()}")
    modified_metadata = modified_metadata.replace(line, b"", 1)
if digest(modified_metadata) != expected_modified_hash:
    raise SystemExit("mlx-audio modified METADATA hash does not match the reviewed overlay")

metadata_record_path = b"mlx_audio-0.4.5.dist-info/METADATA,"
record_lines = original_record.splitlines(keepends=True)
matching_indexes = [
    index for index, line in enumerate(record_lines) if line.startswith(metadata_record_path)
]
if len(matching_indexes) != 1:
    raise SystemExit("mlx-audio RECORD does not contain exactly one METADATA entry")
index = matching_indexes[0]
ending = b"\r\n" if record_lines[index].endswith(b"\r\n") else b"\n"
encoded_hash = base64.urlsafe_b64encode(hashlib.sha256(modified_metadata).digest()).rstrip(b"=")
record_lines[index] = (
    metadata_record_path
    + b"sha256="
    + encoded_hash
    + b"," + str(len(modified_metadata)).encode("ascii")
    + ending
)
modified_record = b"".join(record_lines)

metadata_temp = metadata_path.with_suffix(".tmp")
record_temp = record_path.with_suffix(".tmp")
metadata_temp.write_bytes(modified_metadata)
record_temp.write_bytes(modified_record)
os.replace(metadata_temp, metadata_path)
os.replace(record_temp, record_path)

overlay = {
    "schema_version": 1,
    "package": "mlx-audio",
    "version": "0.4.5",
    "upstream_wheel": {"url": wheel_url, "sha256": wheel_hash},
    "reason": (
        "Noice City prunes the unused SciPy resampler and SoundDevice player; "
        "the release overlay removes only their two unconditional Requires-Dist lines."
    ),
    "removed_requires_dist": ["scipy>=1.10.0", "sounddevice>=0.5.3"],
    "original_metadata": {
        "sha256": digest(original_metadata),
        "size": len(original_metadata),
    },
    "modified_metadata": {
        "sha256": digest(modified_metadata),
        "size": len(modified_metadata),
    },
    "original_record": {"sha256": digest(original_record), "size": len(original_record)},
    "modified_record": {"sha256": digest(modified_record), "size": len(modified_record)},
    "record_entry_updated": "mlx_audio-0.4.5.dist-info/METADATA",
    "package_code_modified": False,
}
(provenance_root / "MLX_AUDIO_METADATA_OVERLAY.json").write_text(
    json.dumps(overlay, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

if ! "$PYTHON" -I -B -m pip check > "$DEPENDENCY_PROVENANCE_ROOT/PIP_CHECK.txt" 2>&1; then
  /bin/cat "$DEPENDENCY_PROVENANCE_ROOT/PIP_CHECK.txt" >&2
  die "pip check failed after the reviewed MLX-Audio metadata overlay"
fi
if ! /usr/bin/grep -Fqx 'No broken requirements found.' "$DEPENDENCY_PROVENANCE_ROOT/PIP_CHECK.txt"; then
  die "pip check produced unexpected output"
fi

# The app has no GUI code in Python and does not use IDLE. Remove the optional
# Tcl/Tk stack rather than shipping unused native libraries and resources.
/bin/rm -rf \
  "$STAGING/runtime/lib/python3.12/tkinter" \
  "$STAGING/runtime/lib/python3.12/idlelib"
/usr/bin/find "$STAGING/runtime/lib/python3.12/lib-dynload" -maxdepth 1 \
  -type f -name '_tkinter*.so' -delete
/usr/bin/find "$STAGING/runtime/lib" -maxdepth 1 -type d \
  \( -name 'tcl*' -o -name 'tk*' -o -name 'itcl*' -o -name 'thread*' \) \
  -prune -exec /bin/rm -rf {} +
/usr/bin/find "$STAGING/runtime/lib" -maxdepth 1 -type f \
  \( -name 'libtcl*' -o -name 'libtk*' -o -name 'libitcl*' -o -name 'libthread*' \) \
  -delete

# pip and dependency console entry points embed the temporary preparation path
# in their shebangs. The launcher uses python3 -m uvicorn, so only interpreter
# entry points belong in the distributable runtime.
while IFS= read -r -d '' entry; do
  case "$(basename "$entry")" in
    python|python3|python3.12) ;;
    *) /bin/rm -rf "$entry" ;;
  esac
done < <(/usr/bin/find "$STAGING/runtime/bin" -mindepth 1 -maxdepth 1 -print0)

# Remove dependency and stdlib test payloads as well as mutable caches.
/usr/bin/find "$STAGING/runtime" \
  \( -name '__pycache__' -o -name '.pytest_cache' -o -name 'test' -o -name 'tests' \) \
  -type d -prune -exec /bin/rm -rf {} +
/usr/bin/find "$STAGING/runtime" \
  \( -name '*.pyc' -o -name '*.pyo' -o -name '.DS_Store' \) \
  -type f -delete

# This bundle-wide deployment-target audit is the release gate that prevents a
# host-specific binary wheel from entering the runtime unnoticed.
"$SCRIPT_DIR/audit-macho.sh" \
  --root "$STAGING/runtime" \
  --maximum-macos "$TARGET_MACOS_VERSION" \
  --required-arch arm64 \
  --report "$DEPENDENCY_PROVENANCE_ROOT/MACHO_COMPATIBILITY.txt"

# Record the final post-prune installed distribution set, then checksum every
# retained dependency input and audit artifact (excluding the checksum file
# itself). These files are promoted into the app's RELEASE_PROVENANCE.
"$PYTHON" -I -B - "$DEPENDENCY_PROVENANCE_ROOT" <<'PY'
from __future__ import annotations

import hashlib
import importlib.metadata
import sys
from pathlib import Path

root = Path(sys.argv[1])
packages: dict[str, str] = {}
for dist in importlib.metadata.distributions():
    name = dist.metadata.get("Name")
    if not name:
        raise SystemExit(f"installed distribution has no Name: {dist._path}")
    canonical = name.lower().replace("_", "-")
    if canonical in packages:
        raise SystemExit(f"duplicate installed distribution: {canonical}")
    packages[canonical] = dist.version
(root / "INSTALLED_PACKAGES.txt").write_text(
    "".join(f"{name}=={packages[name]}\n" for name in sorted(packages)),
    encoding="utf-8",
)

manifest: list[str] = []
for path in sorted(root.iterdir(), key=lambda item: item.name):
    if not path.is_file() or path.name == "SHA256SUMS.txt":
        continue
    manifest.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
(root / "SHA256SUMS.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
PY

"$PYTHON" -I -B - <<'PY'
import importlib.util

import fastapi
import mlx
import mlx_audio
import mlx_lm
import uvicorn
from mlx_audio.stt.utils import load as load_stt
from mlx_audio.tts.utils import load_model as load_tts

for pruned in ("scipy", "sounddevice", "tkinter", "idlelib"):
    if importlib.util.find_spec(pruned) is not None:
        raise SystemExit(f"pruned package remains importable: {pruned}")
print("validated pruned bundled dependencies")
PY

if /usr/bin/find "$STAGING/runtime/bin" -mindepth 1 -maxdepth 1 \
  ! -name python ! -name python3 ! -name python3.12 -print -quit | /usr/bin/grep -q .; then
  die "runtime/bin contains a non-interpreter entry point"
fi
[[ -s "$PROVENANCE_ROOT/PYTHON.json" ]] || die "runtime provenance metadata is missing"
[[ -s "$PROVENANCE_ROOT/ARCHIVE_SHA256S.txt" ]] || die "runtime archive checksums are missing"
[[ -s "$PROVENANCE_ROOT/LICENSES_SHA256.txt" ]] || die "runtime license manifest is missing"
[[ -s "$PROVENANCE_ROOT/SUPPLEMENTAL_LICENSE_PROVENANCE.txt" ]] || die "runtime supplemental license provenance is missing"
for dependency_provenance in \
  uv.lock requirements.txt INSTALLED_PACKAGES.txt PIP_CHECK.txt \
  MLX_AUDIO_METADATA_OVERLAY.json MLX_TARGET_WHEELS.txt MACHO_COMPATIBILITY.txt \
  RUNTIME_PACKAGE_LICENSES_SHA256.txt \
  RUNTIME_PACKAGE_LICENSE_PROVENANCE.txt SHA256SUMS.txt; do
  [[ -s "$DEPENDENCY_PROVENANCE_ROOT/$dependency_provenance" ]] || \
    die "runtime dependency provenance is missing: $dependency_provenance"
done
if ! /usr/bin/find "$RUNTIME_PACKAGE_LICENSE_ROOT" -type f -print -quit | /usr/bin/grep -q .; then
  die "runtime package license set is empty"
fi

/bin/rm -rf "$OUTPUT_DIR"
/bin/mv "$STAGING/runtime" "$OUTPUT_DIR"
echo "$OUTPUT_DIR"
