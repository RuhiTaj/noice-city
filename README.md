<div align="center">
  <img src="assets/branding/NoiceCity-1024.png" width="180" alt="Noice City skyline and soundwave logo">
  <h1>Noice City</h1>
  <p><strong><a href="https://github.com/RuhiTaj/noice-city/releases/download/v0.1.1/Noice-City-0.1.1-arm64.dmg">Download Noice City 0.1.1 for Apple Silicon</a></strong></p>
  <p><a href="LICENSE">Apache License 2.0</a> · <a href="TRADEMARKS.md">Brand policy</a></p>
  <p>Developer ID signed, Apple notarized, and stapled for offline verification.<br>
  <a href="https://github.com/RuhiTaj/noice-city/releases/tag/v0.1.1">Release notes, compatibility audit, and SHA-256 checksum</a></p>
</div>

Noice City creates persistent personal voice variants from the installed
Qwen3-TTS 1.7B Base model on Apple Silicon. Home, Work, Narration, or any other
variant shares the immutable base weights while keeping its recordings, frozen
datasets, LoRA adapter, speaker embedding, evaluations, and promoted checkpoint
independent. Inference loads one variant artifact, installs its embedding in
slot `3000`, switches Qwen to `custom_voice`, and generates from text with no
reference recording.

## Interface preview

<table>
  <tr>
    <td width="50%" valign="top">
      <a href="docs/screenshots/generate.png"><img src="docs/screenshots/generate.png" alt="Noice City reference-free speech generation interface"></a><br>
      <sub><strong>Generate locally:</strong> write text and hear it in a persistent trained voice with reproducible controls.</sub>
    </td>
    <td width="50%" valign="top">
      <a href="docs/screenshots/record.png"><img src="docs/screenshots/record.png" alt="Noice City guided voice recording interface"></a><br>
      <sub><strong>Build a private dataset:</strong> record guided prompts and natural answers entirely on your Mac.</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <a href="docs/screenshots/train.png"><img src="docs/screenshots/train.png" alt="Noice City local voice training and checkpoint interface"></a><br>
      <sub><strong>Train and compare:</strong> freeze datasets, preserve checkpoints, and run blinded A/B evaluations.</sub>
    </td>
    <td width="50%" valign="top">
      <a href="docs/screenshots/voice-variants.png"><img src="docs/screenshots/voice-variants.png" alt="Noice City independent voice variant manager"></a><br>
      <sub><strong>Keep styles separate:</strong> use independent voice lineages for accents, contexts, and delivery styles.</sub>
    </td>
  </tr>
</table>

These screenshots come from a clean, isolated local fixture. They contain no
personal recordings, transcripts, checkpoints, browser chrome, or user data.

All private state is stored under:

```text
~/Library/Application Support/VoiceTrainer
```

This legacy internal path is intentionally preserved after the Noice City
rename so existing recordings, datasets, models, and checkpoints remain linked.

Noice City does not import recordings, transcripts, prompt caches, profiles,
or environments from other voice applications. The first-run setup installs
only the pinned Qwen Base BF16 snapshot required for recording, training, and
generation. A separate, optional action can install the pinned Whisper snapshot
for automatic transcription, transcript comparison/WER checks, and checkpoint
evaluation.

Project policies: [Privacy](PRIVACY.md) · [Security](SECURITY.md) ·
[Contributing](CONTRIBUTING.md) · [Code of Conduct](CODE_OF_CONDUCT.md) ·
[License](LICENSE) · [Notice](NOTICE) ·
[Brand and trademarks](TRADEMARKS.md) ·
[Third-party notices](THIRD_PARTY_NOTICES.md) · [Model notices](MODEL_NOTICES.md)

## Prerequisites

- An Apple Silicon Mac running macOS 14.0 or newer
- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)

## One-time setup

```bash
cd noice-city
./setup.sh
```

This creates the locked development environment. Start Noice City, then use
the explicit first-run screen to download approximately 4.5 GB of pinned Qwen
weights. The downloader runs in its own setup process, verifies and hashes every
installed file, and atomically stores the model in private Application Support
storage. If the exact Qwen snapshot is already in the local Hugging Face cache,
setup adopts it without downloading it again.

Whisper is a separate approximately 1.6 GB install offered inside the app. It
is optional for recording, manual transcript review, and generation. It is
required before starting Noice City's training workflow because every saved
checkpoint automatically runs transcription/WER and checkpoint evaluation.
Speech activity, silence, clipping, speaking-level/noise, and duration checks
run locally without Whisper.

Neither download begins from its first button click. A confirmation dialog first
shows the exact Hugging Face repository, pinned revision, declared model-card
license, approximate size, private destination under
`~/Library/Application Support/VoiceTrainer`, and a link to the model notices
served by the local app. The user must then explicitly confirm that specific
download.

Locked ML stack:

- Python 3.12
- MLX 0.32.0
- MLX-LM 0.31.3
- MLX-Audio 0.4.5

## Start offline

```bash
./start.sh
```

Ordinary startup never installs, resolves packages, or contacts a repository.
It binds only to `http://127.0.0.1:8787`, creates or reuses a private session
secret stored in Application Support, and opens the private browser UI. On a
fresh Mac, network access begins only after the user presses **Download
Qwen3-TTS** or separately chooses to install **Whisper automatic checks**. The
API process and all later recording, training, evaluation, and generation
workers remain offline. API writes require both the session secret and the exact
loopback Origin.

## Tests

```bash
.venv/bin/python -m pytest
```

## Workflow

1. Choose or create a **Voice variant**. Every recording session is pinned to
   that stable lineage, and the selector scopes Record, Review, Train, and
   Generate. Variant display names may be changed without changing ownership.
2. Choose that variant's **Prompt pack** on Record. Fresh installs and new
   variants default to **English (US)** vocabulary, places, currency, dates,
   and everyday terms. **Indian English** remains available for Indian names,
   places, currency, food, and terminology. The selection is stored per voice
   variant and changes text coverage—not the accent the model learns from you.
3. **Record** starts with the recommended automatic mix: 80% guided reading,
   20% interview answers, 90% neutral, and 10% mild expression. Prompts are
   assigned to 80% train, 10% validation, and 10% locked test before capture.
4. Masters are immutable 48 kHz mono PCM24 WAV. Noice City creates immutable
   24 kHz mono PCM16 training derivatives. Its built-in audio analysis checks
   speech activity, silence, clipping, speaking level/noise, and duration
   without Whisper. When the optional Whisper model is installed, it also adds
   automatic transcription, transcript comparison, and WER checks. Otherwise,
   transcript review remains manual. Uncertain takes require complete playback
   and explicit transcript confirmation.
5. After two approved minutes from the built-in Recommended mix, **My script**
   unlocks for that voice variant. Paste up to 20,000 characters and the app
   creates short, resumable recording takes. These are supplemental guided,
   neutral training items; validation/test and coverage gates continue to use
   the preassigned catalog prompts.
6. Install Whisper before starting training; the app requires it for automatic
   checkpoint transcription/WER and evaluation. At two accepted minutes, freeze
   a pilot dataset and train one pilot checkpoint. At thirty accepted minutes,
   freeze a full dataset; sixty minutes is recommended. Training always reads a
   checksummed frozen manifest, never the live recording collection.
7. Training runs in a separate offline process and exits after saving its
   artifacts. A different fresh process reloads each checkpoint and generates
   the twenty locked narration tests without reference audio.
8. Local evaluation measures Whisper WER, Qwen ECAPA speaker similarity against
   held-out real test takes, repetition, silence, termination, and duration.
   Complete the hidden A/B listening review in **Train**. A checkpoint is
   promoted only when all objective gates pass and it wins at least 12/20
   comparisons. Failed candidates and the prior promoted version are retained.
9. **Generate** accepts text, a delivery preset, a checkpoint version, and an
   optional integer seed (`0`–`4294967295`) for reproducible sampling. Omitting
   the seed preserves stochastic generation. The API has no reference-audio
   field.

The pilot and thirty-minute gates apply independently to every variant. Full
training is blocked until that same variant has a promoted pilot. Full runs use
batch size 1, gradient accumulation 4, gradient checkpointing, learning rate
`1e-5`, seed `3407`, and preserve all three epochs. Checkpoint promotion and A/B
comparison are also isolated per variant.

## Local API

The primary routes are:

```text
GET    /api/status
GET    /api/model-setup
POST   /api/model-setup
POST   /api/model-setup/cancel
GET    /api/variants
POST   /api/variants
PATCH  /api/variants/{id}
POST   /api/recording-sessions
POST   /api/custom-scripts
GET    /api/custom-scripts/next
GET    /api/prompts/next
POST   /api/recordings
PATCH  /api/recordings/{id}
POST   /api/recordings/bulk
POST   /api/datasets/freeze
POST   /api/training-runs
GET    /api/training-runs/{id}
DELETE /api/training-runs/{id}
POST   /api/generations
```

Checkpoint evaluation, protected A/B audio, votes, and retries are also exposed
under `/api/checkpoints/{id}/...` for the local UI.

## macOS app and DMG

Noice City 0.1.1 is available from the [GitHub release
page](https://github.com/RuhiTaj/noice-city/releases/tag/v0.1.1). The DMG is
Developer ID signed, Apple notarized, stapled, and published with a SHA-256
checksum. The current package target is Apple Silicon on macOS 14.0 or newer;
Intel Macs are not supported because MLX requires Apple Silicon. Version 0.1.0
was withdrawn after its release build incorrectly selected macOS 26-specific
MLX wheels. The stable package is promoted only after its GitHub-downloaded DMG
passes on a separate macOS 14 Apple Silicon Mac. See the [compatibility policy
and incident audit](docs/COMPATIBILITY.md).

The release scaffold in [`packaging/macos`](packaging/macos/README.md) builds an
Apple Silicon `.app`, signs its nested runtime, creates a drag-to-Applications
DMG, and provides notarization and Gatekeeper verification steps. The bundle
uses an allowlist and never includes `.venv`, model weights, recordings,
datasets, adapters, generated audio, SQLite state, or session tokens.

## Privacy and responsible use

Record or adapt only your own voice, or a voice you have explicit permission to
use. Do not use Noice City to impersonate someone, bypass consent, or mislead an
audience about who is speaking. Voice recordings and trained artifacts are
sensitive biometric-like data; keep Application Support backups private.

Noice City has no cloud speech backend, accounts, analytics, advertising, or
automatic upload path. The one-time model installer contacts Hugging Face only
after an explicit click. Recording, review, training, evaluation, and generation
then run locally. See [PRIVACY.md](PRIVACY.md) for the complete data and network
boundary.

## License

Noice City is open source under the [Apache License 2.0](LICENSE). Third-party
software and model snapshots remain subject to their respective terms; see the
[third-party notices](THIRD_PARTY_NOTICES.md) and [model notices](MODEL_NOTICES.md).
Project attribution is recorded in [NOTICE](NOTICE). The Apache license does not
grant permission to use project trademarks; see the [trademark and brand
policy](TRADEMARKS.md) for the limited permitted uses and fork requirements.

## What exists before recording

The application, training objective, offline workers, evaluation protocol, and
generation runtime are implemented and locally verified. The repository never
contains a real voice adapter or recording; those are created from fresh local
voice data and remain in Application Support. The two-minute pilot is an
intentional quality gate: if it is not recognizably the target speaker, the
objective should be corrected before collecting or training the full set.
