# Noice City model notices

The Apache License 2.0 for Noice City's application code does not relicense any
model weights or user-trained artifacts described here.

No model weights are stored in the Noice City source repository or bundled in
the macOS application/DMG. During an explicit first-run setup, the user
downloads the exact pinned Qwen snapshot used for voice training and
generation. The user may separately add the exact pinned Whisper snapshot for
automatic transcripts and checkpoint speech-quality checks. The application
verifies each pinned revision and records file hashes in a local manifest.
Models, recordings, datasets, adapters, checkpoints, and generated audio
remain in the user's local Application Support directory.

Model licenses and usage terms are controlled by the model publishers. Review
those terms before downloading or using a model. This notice records the
metadata available at the exact revisions used by Noice City; it does not grant
additional rights.

## Qwen3-TTS Base

- Repository: [`mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16`](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16)
- Pinned revision: [`a6eb4f68e4b056f1215157bb696209bc82a6db48`](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16/tree/a6eb4f68e4b056f1215157bb696209bc82a6db48)
- Format used by Noice City: MLX BF16 safetensors
- Approximate download size recorded by the application: 4.54 GB
- License declared by that revision's model-card metadata: Apache-2.0
- Conversion statement in that model card: converted from
  [`Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
  using mlx-audio 0.3.0

The model is downloaded by the user and is not redistributed by Noice City.
See the pinned model card and upstream model card for the authoritative model
description and terms. Qwen is required for recording, voice training, and
generation.

## Optional Whisper speech checks

- Repository: [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo)
- Pinned revision: [`41f01f3fe87f28c78e2fbf8b568835947dd65ed9`](https://huggingface.co/openai/whisper-large-v3-turbo/tree/41f01f3fe87f28c78e2fbf8b568835947dd65ed9)
- Publisher: OpenAI
- Format downloaded by Noice City: Hugging Face FP16 safetensors
- Approximate snapshot size recorded by the application: 1.623 GB
- License declared by that exact revision's model-card metadata: MIT
- Upstream lineage declared by that model card:
  [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3)

Noice City downloads the official publisher snapshot rather than a third-party
conversion. MLX-Audio 0.4.5 loads the Hugging Face Whisper configuration and
weights locally, remaps them into its MLX implementation at runtime, and does
not upload the audio. The model is downloaded by the user and is not
redistributed by Noice City. See the pinned model card for the authoritative
model description and terms. Whisper is not required to record audio or run
local signal-level speech/silence checks; it enables automatic speech-to-text,
transcript comparison, WER, and checkpoint evaluation. Without it, transcript
review remains manual.

## User-trained artifacts

Speaker embeddings, LoRA adapters, datasets, and checkpoints are created from
audio supplied by the user. Noice City does not claim ownership of those
artifacts. Users are responsible for having the rights and consent needed for
the voices and recordings they use.
