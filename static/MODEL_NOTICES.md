# Noice City model notices

No model weights are bundled with Noice City. A model is downloaded only after
the user reviews its exact repository, pinned revision, declared license,
approximate size, and local destination, then explicitly confirms the install.
The application verifies the pinned revision and records file hashes in a local
manifest. Model licenses and usage terms remain controlled by their publishers.

All downloaded models are stored privately under:

```text
~/Library/Application Support/VoiceTrainer
```

## Required Qwen3-TTS Base

- Repository: `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16`
- Pinned revision: `a6eb4f68e4b056f1215157bb696209bc82a6db48`
- Format: MLX BF16 safetensors
- Approximate download size: 4.54 GB
- License declared by the pinned model-card metadata: Apache-2.0
- Converted from: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`

Qwen is required for recording, voice training, and generation. It is
downloaded by the user and is not redistributed by Noice City.

## Optional Whisper automatic checks

- Repository: `openai/whisper-large-v3-turbo`
- Pinned revision: `41f01f3fe87f28c78e2fbf8b568835947dd65ed9`
- Format: Hugging Face FP16 safetensors
- Approximate download size: 1.623 GB
- License declared by the pinned model-card metadata: MIT

Whisper adds automatic speech-to-text, transcript comparison, WER checks, and
checkpoint evaluation. It is optional for recording, manual
transcript review, and generation. Noice City's training workflow requires it
because automatic checkpoint evaluation is mandatory. Speech activity,
silence, clipping, speaking-level/noise, and duration checks run without it.

Neither installer uploads recordings, transcripts, trained artifacts, or
generated audio. After installation, model use remains local and offline.
