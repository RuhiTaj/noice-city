# Contributing

Noice City is in active development. Bug reports, focused design feedback, and
reproducible test cases are welcome.

## Before contributing

- Read [PRIVACY.md](PRIVACY.md), [SECURITY.md](SECURITY.md), and the
  [Code of Conduct](CODE_OF_CONDUCT.md).
- Never submit recordings, transcripts from a real speaker, trained adapters,
  embeddings, checkpoints, generated private audio, local databases, session
  tokens, Apple credentials, or signing files.
- Use only your own voice or a voice you have explicit permission to use.
- Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

The project is licensed under the [Apache License 2.0](LICENSE). Unless you
explicitly state otherwise, contributions intentionally submitted for inclusion
in Noice City are provided under that same license, as described in Section 5.

## Development setup

Noice City currently targets Python 3.12 on Apple Silicon.

```bash
git clone https://github.com/RuhiTaj/noice-city.git
cd noice-city
./setup.sh
.venv/bin/python -m pytest
./start.sh
```

The model pack is approximately 6 GB and is installed separately through the
first-run UI. Most unit tests do not require model weights or voice recordings.

## A good issue includes

- the commit tested;
- the macOS version and Mac model or chip;
- exact steps to reproduce the problem;
- expected and observed behavior; and
- sanitized logs or screenshots when useful.

Replace usernames and absolute home paths, and confirm that attachments contain
no voice data or secrets before posting them publicly.

## Change expectations

Keep changes narrow and preserve the core invariants:

- normal application processes remain offline and loopback-only;
- model network access requires an explicit setup action;
- generation never requires reference audio;
- recordings and artifacts remain isolated by voice variant;
- training consumes immutable frozen datasets; and
- tests never depend on personal data or credentials.

Run the full test suite before requesting review:

```bash
.venv/bin/python -m pytest
```

Documentation should distinguish implemented behavior from planned behavior.
Do not add a download link until the linked artifact is signed, notarized,
stapled, checksummed, and verified on a clean compatible Mac.
