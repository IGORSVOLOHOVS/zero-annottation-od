# Manual

`zeod` turns a folder of unlabelled helmet photos into a trained YOLOv8 detector in four steps. Every step reads `configs/default.yaml`; `--config <file>` and `--set key.path=value` go before the subcommand.

| step | command | reads | writes |
| --- | --- | --- | --- |
| label | `.\recipe.ps1 run label [--limit N]` | `images/` | `labels_raw.json`, checkpointed every 20 images |
| build the dataset | `.\recipe.ps1 run build-dataset` | `labels_raw.json` | `dataset/{train,val}/{images,labels}` and `data.yaml` |
| train | `.\recipe.ps1 run train` | `dataset/` | `runs/` with the weights |
| evaluate | `.\recipe.ps1 run evaluate [--analyze-errors]` | the weights and `dataset/val` | metrics and false positive / false negative examples |

`.\recipe.ps1 run pipeline [--limit N]` runs all four in order; `--limit 10` is the smoke test. `.\recipe.ps1 run infer "<glob>" --output <folder>` draws the trained detector's boxes on new images.

`label` starts `llama-server` itself, waits until it answers and restarts it when it stops responding. It resumes from `labels_raw.json`, so an interrupted run loses at most one checkpoint.

`benchmarks/` measures the labeling helpers; run it with `uv run --frozen python -m pytest benchmarks --no-cov`. `docs/quality-iso25010.md` holds the ISO 25010 assessment and `docs/architecture.md` the module map.
