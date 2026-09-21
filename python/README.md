# Analysis engine

Python package for the local golf swing analyzer. Run commands from the repository
root so paths match the examples:

```bash
uv sync --project python --locked
uv run --project python analyzer doctor
uv run --project python analyzer --help
uv run --project python analyzer extract path/to/swing.mov
uv run --project python analyzer metrics path/to/swing.mov
```

The CLI and desktop worker share `analyzer.dispatch`; Pydantic models in
`analyzer/contracts/` define their data. `uv run --project python python -m analyzer.worker`
starts the newline-delimited JSON-RPC process used by Rust. Stdout is reserved
for the protocol; diagnostics go to stderr.

See the root [README](../README.md) for setup, [architecture](../docs/architecture.md)
for process and cache boundaries, [testing](../docs/testing.md) for checks, and
[modeling](../docs/modeling.md) for model installation and training. The installed
Python minor version and exact dependency resolution are pinned by
`pyproject.toml` and `uv.lock`.
