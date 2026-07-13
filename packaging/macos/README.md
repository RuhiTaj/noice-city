# Noice City macOS distribution scaffold

This directory builds an Apple Silicon `Noice City.app` and a drag-to-install
DMG without touching the currently running development server. The application
bundle contains the app code and a relocatable Python/MLX runtime. It does **not**
contain a developer virtual environment, Qwen or Whisper weights, recordings,
transcripts, datasets, checkpoints, database files, session tokens, or caches.

Each installation owns its mutable data under:

```text
~/Library/Application Support/VoiceTrainer
~/Library/Caches/VoiceTrainer
~/Library/Logs/VoiceTrainer
```

Those internal directories intentionally keep the `VoiceTrainer` name so an
existing local installation, recordings, datasets, and checkpoints continue to
work after the Noice City rebrand.

Model downloads are therefore explicit and separate. Qwen is the required
approximately 4.5 GB first-run voice model for recording, training, and
generation. Whisper is an optional approximately 1.6 GB install for automatic
transcription, transcript comparison/WER checks, and checkpoint evaluation.
Built-in speech-activity, silence, clipping, speaking-level/noise, and duration
checks do not need Whisper. Noice City's training endpoint does require Whisper
because automatic checkpoint evaluation is a mandatory part of every training
run. Once installed in Application Support, ordinary app startup forces model
libraries offline and binds the server only to `127.0.0.1:8787`.

## Prerequisites

- Apple Silicon Mac running macOS 14.0 or newer. The release builder installs
  the exact hash-pinned `macosx_14_0_arm64` MLX and MLX-Metal wheels before the
  rest of the locked graph. Intel Macs are unsupported because MLX requires
  Apple Silicon.
- Xcode command-line tools.
- `uv` for preparing the release runtime.
- `zstd` for extracting the pinned full Python archive (for example,
  `brew install zstd`).
- The exact matching `install_only` and `full`
  [python-build-standalone](https://github.com/astral-sh/python-build-standalone)
  CPython 3.12.13 arm64 archives. The script verifies these pinned SHA-256s:

  ```text
  3724aa4dafb5f7b6c2cf98e89914e4248dc6bd2fe40407df4a2d73de99615f16  cpython-3.12.13+20260623-aarch64-apple-darwin-install_only.tar.gz
  ce5a2d552077d869f69dc25d834c2fe4d036f9d78e770fb5d916273db802cabc  cpython-3.12.13+20260623-aarch64-apple-darwin-pgo+lto-full.tar.zst
  ```

  The full archive is release evidence: it supplies `PYTHON.json` and all
  component `license_path` files that the install-only archive omits. One
  upstream metadata defect is handled explicitly: the archive references but
  omits `LICENSE.zlib-ng.txt`, so this repository carries an exact hash-pinned
  copy of `LICENSE.md` from the archive's pinned zlib-ng 2.2.4 source. Its URL,
  source-archive hash, and license hash are written into the app's release
  provenance. The distributable runtime still comes only from `install_only`.
- The branded Finder icon at `assets/branding/NoiceCity.icns`. The builder uses
  it automatically. If that source file is absent, the app remains buildable
  with the generic application icon; `--icon /path/to/another.icns` overrides
  the default for a particular build.
- For public distribution, an Apple Developer ID Application certificate and a
  notarization keychain profile.

Do not pass `.venv` to the builder. Python virtual environments record absolute
paths and are not portable after a DMG installation.

## 1. Prepare the bundled runtime

From the repository root:

```bash
packaging/macos/prepare-runtime.sh \
  --python-archive /absolute/path/cpython-3.12.13+20260623-aarch64-apple-darwin-install_only.tar.gz \
  --python-full-archive /absolute/path/cpython-3.12.13+20260623-aarch64-apple-darwin-pgo+lto-full.tar.zst \
  --output build/macos/runtime
```

This verifies both archive hashes, proves every install-only file matches its
counterpart in the full archive, collects the referenced Python component
licenses, unpacks the relocatable interpreter, and installs the locked
application dependencies using `--only-binary=:all:` and `--require-hashes`.
The platform-sensitive MLX wheels are installed from exact macOS 14 URLs and
hashes before the rest of the graph, then their installed `WHEEL` tags are
verified. Every Mach-O in the resulting runtime is scanned and the build fails
if any architecture slice requires a macOS release newer than 14.0.
The exact `uv.lock` and exported hashed requirements are retained with the
release provenance. The script removes unused Tcl/Tk, SciPy, sounddevice,
tests, caches, and temporary-path console scripts after a smoke import. It may
access package indexes while installing that locked, wheel-only dependency
graph during this release-build step. It never downloads model weights.

Public app builds also require a clean Git worktree and retain the exact source
commit and tree identifiers in `RELEASE_PROVENANCE/SOURCE_COMMIT.txt`.

MLX-Audio 0.4.5 declares SciPy and SoundDevice unconditionally even though
Noice City does not call its optional resampler or audio-player paths. After
those packages are pruned, the preparation script applies a deliberately
narrow metadata overlay: it removes exactly those two `Requires-Dist` lines
from MLX-Audio's `METADATA`, updates the matching `RECORD` entry, and changes no
package code. The upstream wheel hash and the original/modified `METADATA` and
`RECORD` hashes are retained in `MLX_AUDIO_METADATA_OVERLAY.json`. A successful
`pip check` result and the final installed-package inventory are retained too.

The tokenizers 0.22.2 and sentencepiece 0.2.2 macOS wheels omit their upstream
Apache-2.0 texts, so exact hash-verified copies from their pinned source
distributions are promoted into `THIRD_PARTY_LICENSES/runtime-packages`. tqdm's
exact wheel-provided `LICENCE` is promoted alongside them. Source and wheel
hashes plus a license checksum manifest are included in release provenance.

## 2. Build and verify the app

For a local ad-hoc test build:

```bash
packaging/macos/build-app.sh \
  --runtime build/macos/runtime \
  --bundle-id com.example.noicecity \
  --version 0.1.1 \
  --build 3 \
  --sign -

packaging/macos/verify-app.sh 'dist/macos/Noice City.app'
open 'dist/macos/Noice City.app'
```

Opening the built app while the development app is already listening on port
8787 does not stop or replace it. The launcher opens the existing private local
session and exits.

For a public build, use the installed Developer ID identity:

```bash
packaging/macos/build-app.sh \
  --runtime build/macos/runtime \
  --version 0.1.1 \
  --build 3 \
  --sign "Developer ID Application: Ruhitaj Reddypalli (28DPUBPSZD)"

packaging/macos/verify-app.sh --release 'dist/macos/Noice City.app'
```

`--release` is intentionally stricter than local verification. It pins the
public bundle identifier, product name, version/build, minimum macOS version,
canonical icon, Developer ID team and authority, hardened runtime, timestamp,
and empty entitlement set (including no `allow-jit`). It also requires strict
and deep code-signature verification.

The builder uses an explicit allowlist (`voice_trainer`, `static`,
`assets/prompts.json`, and `pyproject.toml`) and scans the resulting bundle for
common private artifacts and release-machine paths. Runtime licenses are placed
under `Contents/Resources/THIRD_PARTY_LICENSES` and archive provenance under
`Contents/Resources/RELEASE_PROVENANCE`. The latter includes the pinned
Python-archive evidence and a `runtime-dependencies` directory containing the
exact lockfile, hashed requirements, installed-package inventory, `pip check`
result, MLX-Audio overlay audit, and runtime-package license provenance.
The runtime is validated as Python 3.12 arm64 before the app is emitted.
The canonical `assets/branding/NoiceCity.icns` artwork is copied to
`Contents/Resources/AppIcon.icns` and registered in the bundle automatically.

## 3. Create the DMG

```bash
packaging/macos/build-dmg.sh \
  --app 'dist/macos/Noice City.app' \
  --output dist/macos/Noice-City-0.1.1-arm64.dmg \
  --volume-name 'Noice City' \
  --release \
  --sign "Developer ID Application: YOUR NAME (TEAMID1234)"
```

`--release` runs the complete public-release verifier on both the source app
and the copy mounted from the finished DMG. It also rejects an ad-hoc signing
identity.

The mounted DMG contains the app and an `Applications` symlink for conventional
drag-to-install behavior.

## 4. Notarize and staple

Store notarization credentials once (outside the repository):

```bash
xcrun notarytool store-credentials Noice-City-notary \
  --apple-id you@example.com \
  --team-id TEAMID1234
```

Enter the app-specific password at the secure prompt. Do not put it directly
on the command line, where it can be retained in shell history or exposed to
other local processes.

Submit the signed DMG:

```bash
  packaging/macos/notarize.sh \
  --dmg dist/macos/Noice-City-0.1.1-arm64.dmg \
  --keychain-profile Noice-City-notary
```

The script waits for Apple, staples the ticket, validates it, and runs a
Gatekeeper assessment.

## First-run model flow

The packaging layer deliberately does not put model files in the `.app`. The
application setup screen and `/api/model-setup` service:

1. Explain the exact required Qwen repository, pinned revision, declared
   model-card license, approximately 4.5 GB size, and private Application
   Support destination before connecting. The app becomes usable after this
   model alone is verified.
2. Offer Whisper separately and label it optional for recording, manual review,
   and generation, while clearly stating it must be installed before training
   because checkpoint transcription/WER evaluation is automatic and mandatory.
3. Explain that speech activity, silence, clipping, speaking-level/noise, and
   duration checks continue to work without Whisper.
4. Link to a locally served `MODEL_NOTICES.md`, then require a second explicit
   confirmation in the disclosure dialog before either model install begins.
5. Download to a private staging directory under Application Support.
6. Reject unsafe file types, verify the pinned revision and every expected hash,
   then atomically install the completed model directory.
7. Delete partial staging data on failure or cancellation.
8. Restore all Hugging Face offline and telemetry-disabled environment settings
   before starting inference, recording QA, training, evaluation, or generation.
9. Offer retry after a failed or incomplete install without ever resolving
   packages or contacting a model repository during ordinary startup.

The `.app` launcher and normal API process remain offline. Only the dedicated
installer subprocess receives explicit Hub access while the setup screen is
active.

## Release checklist

- Confirm the chosen bundle identifier and publisher copyright.
- Verify the canonical `assets/branding/NoiceCity.icns` artwork in Finder, the
  Dock, Launchpad, and the installer DMG.
- Record checksums and licenses for the Python runtime and all bundled packages.
- Confirm `LICENSE`, `NOTICE`, `TRADEMARKS.md`, `THIRD_PARTY_NOTICES.md`, and
  `MODEL_NOTICES.md` are present in the application bundle.
- Produce a clean runtime from the locked dependency graph.
- Confirm installed MLX wheel tags are exactly `macosx_14_0_arm64` and run the
  exhaustive Mach-O deployment-target audit over the finished app.
- Build with a Developer ID identity; never publish an ad-hoc-signed app.
- Inspect the bundle for local paths and private data.
- Verify on a clean Apple Silicon user account with no developer tools.
- Test first-run download, cancellation, relaunch, and fully offline operation.
- Test record → freeze → train → quit → relaunch → generate.
- Notarize and staple the exact DMG that will be published.
- Publish SHA-256 checksums and verify the downloaded asset before announcing it.
