#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/noice-macho-policy.XXXXXX")"
trap '/bin/rm -rf "$WORK"' EXIT

/usr/bin/printf '%s\n' 'int main(void) { return 0; }' > "$WORK/main.c"
/usr/bin/xcrun --sdk macosx clang -arch arm64 -mmacosx-version-min=14.0 \
  "$WORK/main.c" -o "$WORK/compatible"
/usr/bin/xcrun --sdk macosx clang -arch arm64 -mmacosx-version-min=15.0 \
  "$WORK/main.c" -o "$WORK/too-new"
/usr/bin/xcrun --sdk iphoneos clang -arch arm64 -miphoneos-version-min=17.0 \
  "$WORK/main.c" -o "$WORK/wrong-platform"
/usr/bin/xcrun --sdk macosx clang -arch arm64 -mmacosx-version-min=14.0 \
  -Wl,-rpath,/tmp/noice-unsafe "$WORK/main.c" -o "$WORK/unsafe-rpath"
/bin/cp "$WORK/compatible" "$WORK/unsafe-load"
/usr/bin/install_name_tool -change /usr/lib/libSystem.B.dylib \
  /tmp/noice-unsafe/libSystem.B.dylib "$WORK/unsafe-load"
/bin/cp "$WORK/compatible" "$WORK/relative-load"
/usr/bin/install_name_tool -change /usr/lib/libSystem.B.dylib \
  libSystem.B.dylib "$WORK/relative-load"

mkdir "$WORK/pass" "$WORK/fail" "$WORK/platform-fail" \
  "$WORK/rpath-fail" "$WORK/load-fail" "$WORK/relative-load-fail"
/bin/cp "$WORK/compatible" "$WORK/pass/compatible"
/bin/cp "$WORK/too-new" "$WORK/fail/too-new"
/bin/cp "$WORK/wrong-platform" "$WORK/platform-fail/wrong-platform"
/bin/cp "$WORK/unsafe-rpath" "$WORK/rpath-fail/unsafe-rpath"
/bin/cp "$WORK/unsafe-load" "$WORK/load-fail/unsafe-load"
/bin/cp "$WORK/relative-load" "$WORK/relative-load-fail/relative-load"

"$ROOT/packaging/macos/audit-macho.sh" \
  --root "$WORK/pass" \
  --maximum-macos 14.0 \
  --required-arch arm64 \
  --report "$WORK/report.tsv"

EXPECTED_HEADER=$'path\tarchitectures\tslice\tplatform\tminimum_macos\tsdk'
[[ "$(/usr/bin/head -n 1 "$WORK/report.tsv")" == "$EXPECTED_HEADER" ]]
/usr/bin/awk -F '\t' '
  NR > 1 && $1 == "compatible" && $2 == "arm64" && $3 == "arm64" &&
    $4 == "MACOS" && $5 == "14.0" && $6 != "" { found = 1 }
  END { exit found ? 0 : 1 }
' "$WORK/report.tsv" || {
  /bin/cat "$WORK/report.tsv" >&2
  exit 1
}

if "$ROOT/packaging/macos/audit-macho.sh" \
    --root "$WORK/fail" \
    --maximum-macos 14.0 \
    --required-arch arm64 >"$WORK/failure.log" 2>&1; then
  echo "newer deployment target unexpectedly passed" >&2
  exit 1
fi
/usr/bin/grep -Fq 'requires macOS 15.0; maximum is 14.0' "$WORK/failure.log"

if "$ROOT/packaging/macos/audit-macho.sh" \
    --root "$WORK/platform-fail" \
    --maximum-macos 14.0 \
    --required-arch arm64 >"$WORK/platform-failure.log" 2>&1; then
  echo "non-macOS binary unexpectedly passed" >&2
  exit 1
fi
/usr/bin/grep -Fq "targets 'IOS', expected MACOS" "$WORK/platform-failure.log"

if "$ROOT/packaging/macos/audit-macho.sh" \
    --root "$WORK/rpath-fail" \
    --maximum-macos 14.0 \
    --required-arch arm64 >"$WORK/rpath-failure.log" 2>&1; then
  echo "unsafe absolute LC_RPATH unexpectedly passed" >&2
  exit 1
fi
/usr/bin/grep -Fq \
  "has unsafe LC_RPATH '/tmp/noice-unsafe'" "$WORK/rpath-failure.log"

if "$ROOT/packaging/macos/audit-macho.sh" \
    --root "$WORK/load-fail" \
    --maximum-macos 14.0 \
    --required-arch arm64 >"$WORK/load-failure.log" 2>&1; then
  echo "unsafe absolute dylib load path unexpectedly passed" >&2
  exit 1
fi
/usr/bin/grep -Fq \
  "loads unsafe dylib path '/tmp/noice-unsafe/libSystem.B.dylib'" \
  "$WORK/load-failure.log"

if "$ROOT/packaging/macos/audit-macho.sh" \
    --root "$WORK/relative-load-fail" \
    --maximum-macos 14.0 \
    --required-arch arm64 >"$WORK/relative-load-failure.log" 2>&1; then
  echo "unsafe bare-relative dylib load path unexpectedly passed" >&2
  exit 1
fi
/usr/bin/grep -Fq \
  "loads unsafe dylib path 'libSystem.B.dylib'" \
  "$WORK/relative-load-failure.log"

echo "Mach-O compatibility policy behaves as expected"
