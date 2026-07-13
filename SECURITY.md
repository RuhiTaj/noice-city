# Security policy

Noice City is a loopback-only, local voice-training application. Security
reports are welcome, especially when they concern unintended network access,
exposure of recordings or trained artifacts, session authentication, unsafe
model installation, or the packaged macOS application.

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x macOS release | Yes |
| Current `main` branch | Yes |
| Older commits or modified builds | No |

Security fixes land on the current release line and `main`.

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

Explicit model installation is the exception: after the user presses **Download
Qwen3-TTS**, a dedicated subprocess downloads only the required pinned Qwen
snapshot. A separate **Install Whisper** action downloads the optional snapshot
for automatic transcription, transcript comparison/WER checks, and checkpoint
evaluation. Each subprocess validates required files, records hashes, and
installs its model into private Application Support storage. Basic
speech-activity, silence, clipping, speaking-level/noise, and duration checks do
not need Whisper. Noice City's training workflow does require it because every
checkpoint receives mandatory automatic evaluation. Development setup and
release builds may separately contact Python package indexes when resolving the
already locked dependency graph.

These controls do not replace normal workstation security. Any process running
as the same macOS user may be able to read that user's files. Keep the Mac,
backups, and Apple signing credentials protected, and never publish the contents
of `~/Library/Application Support/VoiceTrainer`.

## Upstream components

If a vulnerability is entirely within Qwen3-TTS, Whisper, MLX, or another
dependency, report it to that upstream project as well. Please still report an
integration issue here when Noice City makes the impact meaningfully different.
