# macOS compatibility policy

Noice City 0.1.1 targets Apple Silicon Macs running macOS 14.0 or newer.
Intel Macs are not supported because the MLX runtime requires Apple Silicon.
The first corrected artifact was published as a release candidate. Stable
promotion requires the exact GitHub-downloaded DMG to pass on a separate,
clean macOS 14 Apple Silicon Mac; the 0.1.1 stable release follows that gate.

## Why 0.1.0 was withdrawn

The 0.1.0 release was signed and notarized correctly, but its dependency
installation ran on a macOS 26 build host without pinning the MLX wheel's
platform tag. Pip selected the valid but host-specific `macosx_26_0_arm64`
MLX and MLX-Metal wheels. The launcher and `Info.plist` also declared 26.2, so
Finder correctly showed the prohibited-app badge on older systems.

The old verifier checked the launcher target and expected 26.2; it did not ask
whether every bundled native binary met the intended product floor. That was a
release-process defect. Lowering only `LSMinimumSystemVersion` would have been
unsafe because three bundled MLX binaries genuinely required macOS 26.2.

## Gates added for 0.1.1

The corrected build:

1. Installs the exact lock-pinned MLX 0.32.0 and MLX-Metal 0.32.0
   `macosx_14_0_arm64` wheels before resolving the remaining locked graph.
2. Verifies both installed `WHEEL` tags and preserves their URLs and SHA-256
   hashes in the app's release provenance.
3. Compiles the native launcher with `-mmacosx-version-min=14.0` and declares
   the same value in `Info.plist`.
4. Scans every architecture slice of every Mach-O in the runtime, finished app,
   and release verifier. The build fails if a file lacks arm64 or any slice has
   a deployment target newer than 14.0.
5. Keeps the bundle Developer ID signed, hardened, notarized, stapled, and
   Gatekeeper-assessed. These trust checks are separate from compatibility.
6. Requires the exact DMG to pass a clean application-data test covering
   record, freeze, train, restart, and reference-free generation before it is
   published.

The release checksum and notarization submission ID are published with each
GitHub release. A GitHub policy check also guards the version, target, exact
wheel hashes, release notices, packaging-script syntax, locked dependency
advisories, and behavioral Mach-O pass/fail fixtures on a macOS 14 runner.
