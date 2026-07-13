# Security policy

Noice City is a loopback-only, local voice-training application. Security
reports are welcome, especially when they concern unintended network access,
exposure of recordings or trained artifacts, session authentication, unsafe
model installation, or the packaged macOS application.

## Supported versions

| Version | Supported |
| --- | --- |
| Current `main` branch | Yes |
| Public macOS DMG | Not released yet |
| Older commits or modified builds | No |

Until the first public release, fixes land on `main`. A versioned support table
will replace this section when downloadable releases exist.

## Report a vulnerability privately

Do not open a public issue containing an exploit, secret, private recording,
local path, or other sensitive material. Use GitHub's private security advisory
form for this repository:

<https://github.com/RuhiTaj/noice-city/security/advisories/new>

Include:

- the affected commit or app version;
- the macOS version and Mac architecture;
- the behavior you observed and the impact you expect;
- minimal reproduction steps; and
- whether the report contains voice data or another sensitive artifact.

Please remove real recordings, model adaptations, session tokens, Apple
credentials, and signing material whenever a synthetic reproduction is enough.
The maintainer will acknowledge reports on a best-effort basis, investigate,
and coordinate disclosure after a fix is available.

## Security boundary

The normal application binds to `127.0.0.1`, requires a private local session
token for API access, checks the loopback Origin on writes, and applies a
same-origin content security policy. Hugging Face and model-library telemetry
are disabled. Ordinary startup, recording, training, evaluation, and generation
are designed to remain offline.

The explicit first-run model installation is the exception: after the user
presses **Download model pack**, a dedicated subprocess downloads only the
pinned Qwen and Whisper snapshots, validates required files, records hashes,
and installs them into private Application Support storage. Development setup
and release builds may separately contact Python package indexes when resolving
the already locked dependency graph.

These controls do not replace normal workstation security. Any process running
as the same macOS user may be able to read that user's files. Keep the Mac,
backups, and Apple signing credentials protected, and never publish the contents
of `~/Library/Application Support/VoiceTrainer`.

## Upstream components

If a vulnerability is entirely within Qwen3-TTS, Whisper, MLX, or another
dependency, report it to that upstream project as well. Please still report an
integration issue here when Noice City makes the impact meaningfully different.
