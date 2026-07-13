#!/usr/bin/env bash
set -euo pipefail

ROOT=""
MAXIMUM_MACOS="14.0"
REQUIRED_ARCH="arm64"
REPORT=""

usage() {
  cat >&2 <<'EOF'
Usage: packaging/macos/audit-macho.sh --root PATH [options]

Options:
  --maximum-macos VERSION  Highest allowed Mach-O deployment target (default: 14.0).
  --required-arch ARCH     Architecture every Mach-O must contain (default: arm64).
  --report FILE            Write a deterministic tab-separated audit report.

The audit visits every regular file below PATH. Every Mach-O must contain the
required architecture, every architecture slice must declare a macOS build
version, and no slice may require a newer macOS release than the configured
maximum. Loaded dylib paths and runtime search paths must also remain portable:
dependencies may use Apple system paths or standard Mach-O path tokens, while
runtime search paths may use loader/executable-relative tokens or system paths.
This intentionally rejects a bundle assembled from host-specific wheels before
it can be signed or published.
EOF
}

die() {
  echo "audit-macho: $*" >&2
  exit 1
}

version_is_at_most() {
  /usr/bin/awk -v actual="$1" -v maximum="$2" 'BEGIN {
    actual_count = split(actual, actual_parts, ".")
    maximum_count = split(maximum, maximum_parts, ".")
    count = actual_count > maximum_count ? actual_count : maximum_count
    for (part_index = 1; part_index <= count; part_index++) {
      left = actual_parts[part_index] + 0
      right = maximum_parts[part_index] + 0
      if (left < right) exit 0
      if (left > right) exit 1
    }
    exit 0
  }'
}

dependency_path_is_safe() {
  case "$1" in
    @rpath|@rpath/*|@loader_path|@loader_path/*|@executable_path|@executable_path/*)
      return 0
      ;;
    /System/Library|/System/Library/*|/usr/lib|/usr/lib/*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

rpath_is_safe() {
  case "$1" in
    @loader_path|@loader_path/*|@executable_path|@executable_path/*)
      return 0
      ;;
    /System/Library|/System/Library/*|/usr/lib|/usr/lib/*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

audit_load_paths() {
  local candidate="$1"
  local architecture="$2"
  local relative="$3"
  local load_commands=""
  local linked_path=""
  local runtime_path=""

  load_commands="$(/usr/bin/otool -arch "$architecture" -l "$candidate" 2>/dev/null)" ||
    die "could not inspect load commands for $relative ($architecture)"

  while IFS= read -r linked_path; do
    [[ -n "$linked_path" ]] || continue
    dependency_path_is_safe "$linked_path" ||
      die "$relative ($architecture) loads unsafe dylib path '$linked_path'"
  done < <(/usr/bin/awk '
    $1 == "cmd" {
      command = $2
      next
    }
    (command == "LC_LOAD_DYLIB" ||
     command == "LC_LOAD_WEAK_DYLIB" ||
     command == "LC_REEXPORT_DYLIB" ||
     command == "LC_LAZY_LOAD_DYLIB" ||
     command == "LC_LOAD_UPWARD_DYLIB") && $1 == "name" {
      line = $0
      sub(/^[[:space:]]*name[[:space:]]+/, "", line)
      sub(/[[:space:]]+\(offset[[:space:]]+[0-9]+\).*$/, "", line)
      print line
      command = ""
    }
  ' <<<"$load_commands")

  while IFS= read -r runtime_path; do
    [[ -n "$runtime_path" ]] || continue
    rpath_is_safe "$runtime_path" ||
      die "$relative ($architecture) has unsafe LC_RPATH '$runtime_path'"
  done < <(/usr/bin/awk '
    $1 == "cmd" {
      command = $2
      next
    }
    command == "LC_RPATH" && $1 == "path" {
      line = $0
      sub(/^[[:space:]]*path[[:space:]]+/, "", line)
      sub(/[[:space:]]+\(offset[[:space:]]+[0-9]+\).*$/, "", line)
      print line
      command = ""
    }
  ' <<<"$load_commands")
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="${2:-}"; shift 2 ;;
    --maximum-macos) MAXIMUM_MACOS="${2:-}"; shift 2 ;;
    --required-arch) REQUIRED_ARCH="${2:-}"; shift 2 ;;
    --report) REPORT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 64 ;;
  esac
done

[[ "$(uname -s)" == "Darwin" ]] || die "must run on macOS"
[[ -d "$ROOT" ]] || die "--root must name an existing directory"
[[ "$MAXIMUM_MACOS" =~ ^[0-9]+([.][0-9]+){1,2}$ ]] || die "invalid --maximum-macos"
[[ "$REQUIRED_ARCH" =~ ^[A-Za-z0-9_]+$ ]] || die "invalid --required-arch"

ROOT="$(cd "$ROOT" && pwd)"
TEMP_REPORT="$(mktemp "${TMPDIR:-/tmp}/noice-macho-audit.XXXXXX")"
trap '/bin/rm -f "$TEMP_REPORT"' EXIT
/usr/bin/printf 'path\tarchitectures\tslice\tplatform\tminimum_macos\tsdk\n' > "$TEMP_REPORT"

MACHO_COUNT=0
SLICE_COUNT=0
while IFS= read -r -d '' candidate; do
  description="$(/usr/bin/file -b "$candidate")"
  [[ "$description" == *Mach-O* ]] || continue
  MACHO_COUNT=$((MACHO_COUNT + 1))

  architectures="$(/usr/bin/lipo -archs "$candidate" 2>/dev/null)" ||
    die "could not inspect architectures: ${candidate#$ROOT/}"
  [[ " $architectures " == *" $REQUIRED_ARCH "* ]] ||
    die "${candidate#$ROOT/} lacks required $REQUIRED_ARCH slice (has: $architectures)"

  for architecture in $architectures; do
    build_info="$(/usr/bin/vtool -arch "$architecture" -show-build "$candidate" 2>/dev/null)" ||
      die "could not inspect build version for ${candidate#$ROOT/} ($architecture)"
    platform="$(/usr/bin/awk '$1 == "platform" { print $2; exit }' <<<"$build_info")"
    minimum_version="$(/usr/bin/awk '$1 == "minos" { print $2; exit }' <<<"$build_info")"
    sdk_version="$(/usr/bin/awk '$1 == "sdk" { print $2; exit }' <<<"$build_info")"
    [[ "$platform" == "MACOS" ]] ||
      die "${candidate#$ROOT/} ($architecture) targets '${platform:-unknown}', expected MACOS"
    [[ -n "$minimum_version" ]] ||
      die "${candidate#$ROOT/} ($architecture) has no macOS deployment target"
    [[ -n "$sdk_version" ]] ||
      die "${candidate#$ROOT/} ($architecture) has no build SDK version"
    version_is_at_most "$minimum_version" "$MAXIMUM_MACOS" ||
      die "${candidate#$ROOT/} ($architecture) requires macOS $minimum_version; maximum is $MAXIMUM_MACOS"
    relative="${candidate#$ROOT/}"
    audit_load_paths "$candidate" "$architecture" "$relative"
    /usr/bin/printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$relative" "$architectures" "$architecture" "$platform" \
      "$minimum_version" "$sdk_version" >> "$TEMP_REPORT"
    SLICE_COUNT=$((SLICE_COUNT + 1))
  done
done < <(/usr/bin/find "$ROOT" -type f -print0)

[[ "$MACHO_COUNT" -gt 0 ]] || die "no Mach-O files found under $ROOT"
[[ "$SLICE_COUNT" -ge "$MACHO_COUNT" ]] || die "not every Mach-O file produced an audited slice"

{
  /usr/bin/head -n 1 "$TEMP_REPORT"
  /usr/bin/tail -n +2 "$TEMP_REPORT" | LC_ALL=C /usr/bin/sort
} > "$TEMP_REPORT.sorted"
/bin/mv "$TEMP_REPORT.sorted" "$TEMP_REPORT"

if [[ -n "$REPORT" ]]; then
  mkdir -p "$(dirname "$REPORT")"
  /bin/cp "$TEMP_REPORT" "$REPORT"
fi

echo "Audited $MACHO_COUNT Mach-O files / $SLICE_COUNT slices: required $REQUIRED_ARCH, maximum macOS $MAXIMUM_MACOS"
