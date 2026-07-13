# Noice City third-party notices

Noice City itself is licensed under the Apache License 2.0; see [LICENSE](LICENSE).
The components below remain subject to their own license and attribution terms.

This document inventories third-party software in the Noice City 0.1.1 macOS
release runtime. It is based on the locked application dependencies and the
locally prepared arm64 release runtime inspected on 2026-07-13. It is not a
replacement for the license text supplied by each project.

The release bundle preserves each package-supplied license file under:

`Noice City.app/Contents/Resources/runtime/lib/python3.12/site-packages/`

The CPython license is preserved at:

`Noice City.app/Contents/Resources/runtime/lib/python3.12/LICENSE.txt`

The exact python-build-standalone component license set and release provenance
used to prepare the interpreter are also promoted to:

`Noice City.app/Contents/Resources/THIRD_PARTY_LICENSES/python-build-standalone/`

`Noice City.app/Contents/Resources/RELEASE_PROVENANCE/python-build-standalone/`

The exact lockfile, hashed requirements, installed-package inventory, package
metadata overlay audit, and promoted runtime-package license hashes are kept at:

`Noice City.app/Contents/Resources/RELEASE_PROVENANCE/runtime-dependencies/`

Exact license texts that were absent or ambiguous at the top level of their
macOS wheels are promoted to:

`Noice City.app/Contents/Resources/THIRD_PARTY_LICENSES/runtime-packages/`

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
| fastapi | 0.139.0 | MIT |
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
| python-multipart | 0.0.32 | Apache-2.0 |
| PyYAML | 6.0.3 | MIT |
| regex | 2026.7.10 | Apache-2.0 AND CNRI-Python |
| rich | 15.0.0 | MIT |
| safetensors | 0.8.0 | Apache-2.0 |
| sentencepiece | 0.2.2 | Apache-2.0 |
| shellingham | 1.5.4 | ISC |
| starlette | 1.3.1 | BSD-3-Clause |
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

The release preparation process also requires the exact companion full archive
`cpython-3.12.13+20260623-aarch64-apple-darwin-pgo+lto-full.tar.zst` (SHA-256
`ce5a2d552077d869f69dc25d834c2fe4d036f9d78e770fb5d916273db802cabc`).
Every entry shipped from the install-only archive is checked against its
counterpart in that full archive. The full archive's `PYTHON.json`, archive
checksums, and license manifest are retained as release provenance.

The component license paths referenced by the pinned `PYTHON.json` cover
CPython, bzip2, Expat, libedit, libffi, liblzma, libuuid, mpdecimal, ncurses,
OpenSSL 1.1, OpenSSL 3, SQLite, Tcl, zlib, and zlib-ng. All 15 texts are copied
into the app's top-level third-party license directory. The pinned full archive
references but accidentally omits its zlib-ng license file, so release
preparation supplies the exact upstream zlib-ng 2.2.4 license and retains its
source/archive hashes in `SUPPLEMENTAL_LICENSE_PROVENANCE.txt`.

### Deliberately pruned optional components

MLX-Audio 0.4.5 declares SciPy and SoundDevice unconditionally for code paths
that Noice City does not call: its SciPy resampler and SoundDevice player. Release
preparation removes SciPy, SoundDevice, PortAudio, and their package metadata
after installing the locked dependency graph. It then removes exactly those
two stale unconditional `Requires-Dist` declarations from MLX-Audio 0.4.5's
installed metadata, updates the corresponding `RECORD` entry, runs `pip check`,
and retains the upstream wheel hash plus the original and modified metadata
hashes. No MLX-Audio package code is changed. SoundFile and libsndfile are not
present in the locked release graph.

The tokenizers 0.22.2 and sentencepiece 0.2.2 macOS wheels omit their upstream
Apache-2.0 texts. Exact copies from their hash-pinned source distributions are
therefore promoted to the top-level runtime-package license directory. tqdm
4.68.4's exact wheel-provided `LICENCE` is promoted there as well. Release
verification checks all three texts against their reviewed SHA-256 values.

The pinned CPython archive contains an optional Tcl/Tk GUI stack. Noice City's
Python process has no GUI code, so release preparation removes `_tkinter`,
`tkinter`, IDLE, and the Tcl/Tk/Itcl/Thread libraries and resources. The Tcl
license text remains in the complete python-build-standalone license set for
conservative attribution. Bundle verification rejects these pruned packages
and libraries if they reappear.

### MPL components

certifi 2026.6.17 is MPL-2.0. tqdm 4.68.4 declares MPL-2.0 AND MIT. Their
package-supplied license files are retained in the bundled runtime.

## No endorsement and warranty

Third-party project names identify their respective works and do not imply
endorsement of Noice City. Third-party software is provided under its own
license and warranty terms. Consult the retained license files for the
authoritative terms.
