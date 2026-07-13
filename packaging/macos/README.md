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

The Qwen and Whisper models are therefore a separate, explicit first-run
download. Once verified and installed in Application Support, ordinary app
startup forces model libraries offline and binds the server only to
`127.0.0.1:8787`.

## Prerequisites

- Apple Silicon Mac running macOS 26.2 or newer. The pinned MLX 0.32.0 wheel
  currently declares that deployment target; supporting older macOS releases
  requires a separately validated MLX build rather than changing the plist.
- Xcode command-line tools.
- `uv` for preparing the release runtime.
- A trusted, checksum-pinned Python 3.12 arm64 `install_only`
  [python-build-standalone](https://github.com/astral-sh/python-build-standalone)
  archive. Keep the archive and its SHA-256 in release provenance; this scaffold
  intentionally does not download an unpinned interpreter for you.
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
  --python-archive /absolute/path/cpython-3.12-arm64-install_only.tar.zst \
  --output build/macos/runtime
```

This unpacks the relocatable interpreter and installs the locked application
dependencies. It may access package indexes while installing the locked
dependency graph during this release-build step. It never downloads model
weights.

## 2. Build and verify the app

For a local ad-hoc test build:

```bash
packaging/macos/build-app.sh \
  --runtime build/macos/runtime \
  --bundle-id com.example.noicecity \
  --version 0.1.0 \
  --build 1 \
  --sign -

packaging/macos/verify-app.sh 'dist/macos/Noice City.app'
open 'dist/macos/Noice City.app'
```

Opening the built app while the development app is already listening on port
8787 does not stop or replace it. The launcher opens the existing private local
session and exits.

For a public build, use the installed Developer ID identity:

```bash
--sign "Developer ID Application: YOUR NAME (TEAMID1234)"
```

Replace the example bundle identifier and signing identity with the values for
the account publishing the build.

The builder uses an explicit allowlist (`voice_trainer`, `static`, `assets`, and
`pyproject.toml`) and scans the resulting bundle for common private artifacts.
The runtime is validated as Python 3.12 arm64 before the app is emitted.
The canonical `assets/branding/NoiceCity.icns` artwork is copied to
`Contents/Resources/AppIcon.icns` and registered in the bundle automatically.

## 3. Create the DMG

```bash
packaging/macos/build-dmg.sh \
  --app 'dist/macos/Noice City.app' \
  --output dist/macos/Noice-City-0.1.0-arm64.dmg \
  --volume-name 'Noice City' \
  --sign "Developer ID Application: YOUR NAME (TEAMID1234)"
```

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
  --dmg dist/macos/Noice-City-0.1.0-arm64.dmg \
  --keychain-profile Noice-City-notary
```

The script waits for Apple, staples the ticket, validates it, and runs a
Gatekeeper assessment.

## First-run model flow

The packaging layer deliberately does not put model files in the `.app`. The
application setup screen and `/api/model-setup` service:

1. Explain the exact Qwen and Whisper downloads, revisions, approximate size,
   and destination before connecting.
2. Begin only after an explicit user click.
3. Download to a private staging directory under Application Support.
4. Reject unsafe file types, verify the pinned revision and every expected hash,
   then atomically install the completed model directory.
5. Delete partial staging data on failure or cancellation.
6. Restore all Hugging Face offline and telemetry-disabled environment settings
   before starting inference, recording QA, training, or generation.
7. Offer retry after a failed or incomplete install without ever resolving
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
- Build with a Developer ID identity; never publish an ad-hoc-signed app.
- Inspect the bundle for local paths and private data.
- Verify on a clean Apple Silicon user account with no developer tools.
- Test first-run download, cancellation, relaunch, and fully offline operation.
- Test record → freeze → train → quit → relaunch → generate.
- Notarize and staple the exact DMG that will be published.
- Publish SHA-256 checksums and verify the downloaded asset before announcing it.
