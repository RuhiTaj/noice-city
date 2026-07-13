# Noice City third-party notices

Noice City itself is licensed under the Apache License 2.0; see [LICENSE](LICENSE).
The components below remain subject to their own license and attribution terms.

This document inventories third-party software in the Noice City 0.1.0 macOS
release runtime. It is based on the locked application dependencies and the
locally prepared arm64 release runtime inspected on 2026-07-13. It is not a
replacement for the license text supplied by each project.

The release bundle preserves package-supplied license files under:

`Noice City.app/Contents/Resources/runtime/lib/python3.12/site-packages/`

The CPython license is preserved at:

`Noice City.app/Contents/Resources/runtime/lib/python3.12/LICENSE.txt`

A rebuild can change transitive dependencies. Release maintainers must compare
this inventory with the actual staged runtime for every release.

## Python runtime packages

The license identifiers below are the SPDX expressions, declared license
fields, or license classifiers shipped in the installed package metadata. A
label here records upstream metadata; it is not an independent legal opinion.

| Component | Version | Declared license metadata |
| --- | ---: | --- |
| annotated-doc | 0.0.4 | MIT |
| annotated-types | 0.7.0 | MIT |
| anyio | 4.14.2 | MIT |
| certifi | 2026.6.17 | MPL-2.0 |
| cffi | 2.1.0 | MIT-0 |
| click | 8.4.2 | BSD-3-Clause |
| fastapi | 0.128.8 | MIT |
| filelock | 3.29.7 | MIT |
| fsspec | 2025.3.0 | BSD-3-Clause |
| h11 | 0.16.0 | MIT |
| hf-xet | 1.5.1 | Apache-2.0 |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| huggingface-hub | 1.23.0 | Apache-2.0 |
| idna | 3.18 | BSD-3-Clause |
| Jinja2 | 3.1.6 | BSD |
| markdown-it-py | 4.2.0 | MIT |
| MarkupSafe | 3.0.3 | BSD-3-Clause |
| mdurl | 0.1.2 | MIT |
| miniaudio | 1.71 | MIT |
| mlx | 0.32.0 | MIT |
| mlx-audio | 0.4.5 | MIT |
| mlx-lm | 0.31.3 | MIT |
| mlx-metal | 0.32.0 | MIT |
| numpy | 2.5.1 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| packaging | 26.2 | Apache-2.0 OR BSD-2-Clause |
| pip | 26.1.2 | MIT; its vendored license files are preserved in the wheel metadata |
| protobuf | 7.35.1 | BSD-3-Clause |
| pycparser | 3.0 | BSD-3-Clause |
| pydantic | 2.13.4 | MIT |
| pydantic-core | 2.46.4 | MIT |
| Pygments | 2.20.0 | BSD-2-Clause |
| python-multipart | 0.0.20 | Apache-2.0 |
| PyYAML | 6.0.3 | MIT |
| regex | 2026.7.10 | Apache-2.0 AND CNRI-Python |
| rich | 15.0.0 | MIT |
| safetensors | 0.8.0 | Apache-2.0 |
| scipy | 1.18.0 | BSD-3-Clause; see native libraries below |
| sentencepiece | 0.2.2 | Apache-2.0 |
| shellingham | 1.5.4 | ISC |
| sounddevice | 0.5.5 | MIT; see PortAudio below |
| soundfile | 0.13.1 | BSD-3-Clause; see libsndfile below |
| starlette | 0.52.1 | BSD-3-Clause |
| tokenizers | 0.22.2 | Apache-2.0 |
| tqdm | 4.68.4 | MPL-2.0 AND MIT |
| transformers | 5.12.1 | Apache-2.0 |
| typer | 0.26.8 | MIT |
| typing-inspection | 0.4.2 | MIT |
| typing-extensions | 4.16.0 | PSF-2.0 |
| uvicorn | 0.39.0 | BSD-3-Clause |

Package source distributions and project metadata are available from each
package's versioned page at `https://pypi.org/project/<name>/<version>/`.

## Native and bundled components requiring particular attention

### CPython and python-build-standalone

The inspected runtime contains CPython 3.12.13 from the local archive
`cpython-3.12.13+20260623-aarch64-apple-darwin-install_only.tar.gz` (SHA-256
`3724aa4dafb5f7b6c2cf98e89914e4248dc6bd2fe40407df4a2d73de99615f16`).
CPython's Python Software Foundation license is retained in the app bundle.
The archive was produced by
[python-build-standalone](https://github.com/astral-sh/python-build-standalone).

The inspected `install_only` archive contains Tcl/Tk dynamic libraries but no
separate Tcl/Tk notice file; its only non-package top-level license file is the
CPython license. Before distributing a public DMG, the release maintainer must
obtain and bundle the corresponding runtime dependency notices (or use a
runtime artifact that already supplies a complete notice set). This document
does not attempt to infer missing build provenance.

### libsndfile

The SoundFile wheel contains `libsndfile_arm64.dylib`; runtime inspection
reports libsndfile 1.2.2. The installed wheel identifies libsndfile as LGPL and
retains the complete LGPL-2.1 text. That text is located at
`site-packages/_soundfile_data/COPYING`. Source is available from the
[libsndfile project](https://github.com/libsndfile/libsndfile/releases/tag/1.2.2).

### PortAudio

The SoundDevice wheel contains `libportaudio.dylib`. Runtime inspection reports
`PortAudio V19.7.0-devel, revision unknown`. The wheel's own README attributes
PortAudio to Ross Bencina and Phil Burk under the MIT License and identifies
the binary source project as
[spatialaudio/portaudio-binaries](https://github.com/spatialaudio/portaudio-binaries).
The exact PortAudio source revision is not recorded in the installed wheel;
release maintainers must archive or otherwise document the corresponding
source before redistribution.

### SciPy GCC runtime libraries

The installed SciPy wheel contains these dynamically linked libraries:

- `libgfortran.5.dylib` and `libgcc_s.1.1.dylib` — GPL-3.0-or-later WITH
  GCC-exception-3.1.
- `libquadmath.0.dylib` — LGPL-2.1-or-later.

SciPy's installed `LICENSE.txt` retains the full applicable license texts and
GCC Runtime Library Exception. The wheel metadata does not identify the exact
GCC source revision used for these binaries; that source provenance should be
archived as part of the public binary release process.

### MPL components

certifi 2026.6.17 is MPL-2.0. tqdm 4.68.4 declares MPL-2.0 AND MIT. Their
package-supplied license files are retained in the bundled runtime.

## No endorsement and warranty

Third-party project names identify their respective works and do not imply
endorsement of Noice City. Third-party software is provided under its own
license and warranty terms. Consult the retained license files for the
authoritative terms.
