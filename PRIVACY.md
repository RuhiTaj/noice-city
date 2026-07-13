# Privacy

Noice City is designed for private, on-device voice adaptation. It does not use
a cloud speech service and does not require a Noice City account.

## Data created on this Mac

Depending on the features used, Noice City stores:

- microphone recordings and local training derivatives;
- prompt assignments, transcripts, review decisions, and quality measurements;
- voice-variant profiles and frozen dataset manifests;
- LoRA adapters, speaker embeddings, checkpoints, and evaluation results;
- generated WAV files;
- the required pinned Qwen snapshot and, when chosen, the optional pinned
  Whisper snapshot; and
- a local state database, logs, caches, and a private session secret.

Mutable application data is kept under the following legacy internal paths:

```text
~/Library/Application Support/VoiceTrainer
~/Library/Caches/VoiceTrainer
~/Library/Logs/VoiceTrainer
```

The `VoiceTrainer` name is intentionally retained so the Noice City rename does
not disconnect existing local data.

## What leaves the Mac

Noice City has no application analytics, advertising, crash-reporting service,
account system, cloud training backend, or automatic voice upload path.

Network access is limited to setup actions initiated by the user:

- `./setup.sh` may contact Python package indexes to install the locked
  development dependencies.
- Pressing **Download Qwen3-TTS** starts a dedicated installer that contacts
  Hugging Face for the required pinned Qwen snapshot. A separate explicit
  **Install Whisper** action downloads the optional snapshot used for automatic
  transcription, transcript comparison/WER checks, and checkpoint evaluation.
  Neither action uploads recordings, transcripts, trained artifacts, or
  generated audio.

After a selected model is verified and installed, ordinary startup, recording,
transcription, training, evaluation, and generation are designed to operate
offline. Basic speech-activity, silence, clipping, speaking-level/noise, and
duration checks do not use Whisper. Noice City binds its UI and API only to
`127.0.0.1`, not to a LAN address.

Operating-system services, the browser, package managers, Git, and other tools
used alongside Noice City have their own privacy behavior and are outside this
application's boundary.

## Your control

Recording and training begin only through explicit local actions. Recordings
remain grouped by voice variant, approved takes are frozen into checksummed
dataset versions, and checkpoints remain local unless the user deliberately
copies them elsewhere.

To remove all Noice City data, quit the application and delete the three
`VoiceTrainer` directories listed above. This permanently removes models,
recordings, datasets, trained voices, generated files, and application state;
make a private backup first if anything must be retained. Removing the app or
source checkout alone does not remove Application Support data.

## Sensitive voice data

Treat recordings, speaker embeddings, and trained adapters as sensitive
biometric-like material. Use only voices you own or have explicit permission to
record and adapt. Do not attach real voice data, local databases, checkpoints,
or session tokens to public GitHub issues.

For a suspected privacy or security flaw, follow [SECURITY.md](SECURITY.md).
