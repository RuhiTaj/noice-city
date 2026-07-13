# Noice City model notices

The Apache License 2.0 for Noice City's application code does not relicense any
model weights or user-trained artifacts described here.

No model weights are stored in the Noice City source repository or bundled in
the macOS application/DMG. During an explicit first-run setup, the user may
choose to download two exact, pinned model snapshots directly from Hugging
Face. The application verifies the pinned revision and records file hashes in
a local manifest. Models, recordings, datasets, adapters, checkpoints, and
generated audio remain in the user's local Application Support directory.

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
description and terms.

## Whisper ASR conversion

- Repository: [`mlx-community/whisper-large-v3-turbo-asr-fp16`](https://huggingface.co/mlx-community/whisper-large-v3-turbo-asr-fp16)
- Pinned revision: [`624c19c9af5603fa73b83bce14d4aeea96156d18`](https://huggingface.co/mlx-community/whisper-large-v3-turbo-asr-fp16/tree/624c19c9af5603fa73b83bce14d4aeea96156d18)
- Format used by Noice City: MLX FP16 safetensors
- Approximate download size recorded by the application: 1.50 GB
- Conversion statement in that model card: converted using mlx-audio 0.2.10

### Unresolved licensing metadata

The exact pinned Whisper conversion's model card has no `license` field and no
license file in its downloaded snapshot. Its stated source link is the
malformed relative identifier `./whisper-large-v3-turbo`, so the source model
and inherited terms cannot be established solely from the pinned snapshot.

Accordingly, this notice does **not** characterize that conversion as MIT,
Apache-2.0, or any other license. Noice City does not bundle or redistribute
the weights, but maintainers should obtain an explicit license/provenance
statement from the publisher or replace the conversion with an independently
validated model whose exact revision declares clear terms before presenting a
public binary release as fully license-cleared.

## User-trained artifacts

Speaker embeddings, LoRA adapters, datasets, and checkpoints are created from
audio supplied by the user. Noice City does not claim ownership of those
artifacts. Users are responsible for having the rights and consent needed for
the voices and recordings they use.
