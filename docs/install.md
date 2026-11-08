# Install

1. Install [uv](https://docs.astral.sh/uv/) and Python 3.13.
2. Clone this repository and run `uv sync --frozen` at its root. It installs PyPI's CPU build of torch; the README shows how a GPU user swaps in the CUDA build.
3. Download a llama.cpp Windows release from https://github.com/ggml-org/llama.cpp/releases and put `llama-server.exe` on `PATH`, or name it in `configs/default.yaml` under `llama_cpp.server_binary`. Only `label` and `pipeline` need it; the model downloads itself on the first run.
4. Put the raw photos in `images/`.

Check the result with `.\recipe.ps1 run --help`: it lists the subcommands.

`.\recipe.ps1 build` packs `src/zeod` into a wheel in `dist\`. The wheel installs a `zeod` command and needs llama.cpp just the same.

`.\recipe.ps1 test` runs the suite, mypy and ruff. It needs no GPU, no network and no llama.cpp.
