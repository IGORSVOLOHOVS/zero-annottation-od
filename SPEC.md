# Software Requirements Specification - Zero-Annotation-OD (`zeod`)

This specification was reverse-engineered from the code at commit `52ceef6` (`origin/dev`) on 2026-11-18.
Each requirement records behaviour that the code implements, the file it comes from, and the test that checks
it, if one exists. Claims found only in documents are listed separately. Paths are relative to the repository root.

<!-- jira-sync:start -->
Jira: KAN-76

## Tasks

This list is one ticket in Jira project KAN, [KAN-76](https://igorsvolohovs.atlassian.net/browse/KAN-76). Each line is a checkbox there: tick it in Jira and the sync ticks it here; add a checkbox there and it is appended here. Done items are the features described in section 3; open items are what the project declares but the code does not do yet.

- [x] Configuration (section 3.1) <!-- spec-configuration -->
- [x] Command-line interface (section 3.2) <!-- spec-command-line-interface -->
- [x] Auto-labeling (section 3.3) <!-- spec-auto-labeling -->
- [x] Dataset building (section 3.4) <!-- spec-dataset-building -->
- [x] Training, evaluation, inference (section 3.5) <!-- spec-training-evaluation-inference -->
- [x] Repository tooling (section 3.6) <!-- spec-repository-tooling -->
- [ ] Device selection picks "CUDA, MPS or CPU" (declared, not implemented) <!-- spec-device-selection-picks-cuda-mps-or-cpu -->
- [ ] Switching to MiniCPM-V or Moondream2 needs only `hf_repo` and the prompt; "the rest of the pipeline ... is model-agnostic" (declared, not implemented) <!-- spec-switching-to-minicpm-v-or-moondream2-needs-only-hf -->
- [ ] Self-consistency labeling (2-3 passes, vote) and two-stage person-then-head grounding (declared, not implemented) <!-- spec-self-consistency-labeling-2-3-passes-vote-and-two- -->
- [ ] `.github/workflows/branch-policy.yml`, weekly run (declared, not implemented) <!-- spec-github-workflows-branch-policy-yml-weekly-run -->
- [ ] Scheduled secret scan; ruff `flake8-bandit` rules (declared, not implemented) <!-- spec-scheduled-secret-scan-ruff-flake8-bandit-rules -->
- [ ] gitleaks and `detect-private-key` in pre-commit; ruff `S`; `requirements-dev.lock`; `ci.yml` (declared, not implemented) <!-- spec-gitleaks-and-detect-private-key-in-pre-commit-ruff -->
- [ ] Coverage `fail_under` 85; ruff `C90` complexity limit 10 (declared, not implemented) <!-- spec-coverage-failunder-85-ruff-c90-complexity-limit-10 -->
- [ ] Profiling and screenshot capture of the application (declared, not implemented) <!-- spec-profiling-and-screenshot-capture-of-the-applicatio -->

<!-- jira-sync:end -->

## 1. Purpose and scope

The system produces a two-class (`helmet`, `no_helmet`) YOLOv8 object detector from unlabeled photos. A local
vision-language model labels the photos, and no person draws any boxes (`README.md`, `src/zeod/cli.py`).

In scope:

- auto-labeling images through a locally started llama.cpp `llama-server` (Qwen2.5-VL by default);
- parsing and repairing the model's text output into pixel bounding boxes;
- converting the boxes into a YOLO dataset with a seeded train/val split;
- fine-tuning, validating and running a YOLOv8 model through Ultralytics;
- a CLI with one subcommand per stage and a YAML configuration;
- repository tooling: CI, pre-push check, branch policy, release build, quality metrics.

Out of scope (not implemented on `dev`):

- the original vLLM-based flow, which survives only as reference material in `legacy/` (`README.md`, "What's in `legacy/`");
- any GUI, service, API server or deployment of the trained detector;
- multi-pass or self-consistency labeling (described in `README.md` as untested);
- the course format of `docs/task.txt` (one notebook holding all code, vLLM on a VM). The project departs from
  it on purpose (`README.md`, "Why this exists").

## 2. Context

Actors:

| Actor | Interaction | Source |
| --- | --- | --- |
| Operator (developer running the pipeline) | Runs `zeod` subcommands from a terminal; reads JSON output and logs; inspects images in the notebook | `src/zeod/cli.py`, `notebooks/eda_and_error_analysis.ipynb` |
| Contributor | Runs tests, ruff, the pre-push check; pushes to `dev` | `CONTRIBUTING.md`, `scripts/check_before_push.py` |
| CI (GitHub Actions) | Lint, test, secret scan, branch policy; release build on tags | `.github/workflows/python-ci.yml`, `.github/workflows/release.yml` |

Who the end users of the detector are is not stated (see section 9).

External systems:

| System | Use | Source |
| --- | --- | --- |
| llama.cpp `llama-server` binary | Child process; HTTP on `127.0.0.1:8090` by default | `src/zeod/labeling/backend.py:79-199`, `configs/default.yaml:40-58` |
| Hugging Face (through `llama-server -hf`) | Model and mmproj download on first run, as described in `README.md` | `backend.py:100-101` |
| Ultralytics YOLO / PyTorch | Training, validation, prediction; CUDA check | `src/zeod/train.py`, `evaluate.py`, `infer.py`, `device.py` |
| GitHub (Actions, Releases) | CI and release publishing | `.github/workflows/` |

Runtime environment:

- Python `>=3.10,<3.14` (`pyproject.toml:10`). CI tests 3.12 on `ubuntu-latest` (`python-ci.yml:20-27`). Release
  builds run on `windows-latest` and `ubuntu-latest` with 3.12 (`release.yml:23-28`).
- `README.md` targets Windows: a PowerShell install and a llama.cpp Windows release. Apart from the error text in
  `backend.py:116-121`, nothing in `src/` is OS-specific (inference from reading `src/`).
- GPU optional. CUDA device 0 is used when available, otherwise CPU (`device.py:10-24`). `README.md` reports a run
  on an RTX 3060 Ti with 8 GB.

## 3. Functional requirements

### 3.1 Configuration

- **FR-1** The system shall load configuration from the YAML file given by `--config`, or from
  `configs/default.yaml` when none is given, and fail with `FileNotFoundError` if the file does not exist.
  Source: `src/zeod/config.py:148-158`, `src/zeod/config.py:11-12`. Verification: `test_default_config_loads`; manual check `--config nonexistent.yaml` -> exit 1.
- **FR-2** The system shall apply each `--set key.path=value` override to the loaded data before validation.
  Values are tried as int, then float, then `true`/`false` as bool, then `null`/`none` as None, and are otherwise
  kept as strings.
  Source: `config.py:128-145`, `config.py:161-165`. Verification: `test_set_override_numeric`, `test_set_override_float`, `test_set_override_null_clears_optional_field`. Despite its name, `test_set_override_bool` overrides `llama_cpp.n_gpu_layers=-1` (an int); bool coercion has no test.
- **FR-3** The system shall reject an override without `=` with `ValueError`.
  Source: `config.py:162-163`. Verification: `test_set_override_invalid_format_raises`; manual check -> exit 1.
- **FR-4** The system shall validate the merged configuration against typed models, and `labeling.system_prompt`
  and `labeling.user_prompt` shall be required.
  Source: `config.py:15-121`, `config.py:167`. Verification: `test_default_config_loads` (indirect).
- **FR-5** The system shall resolve relative paths in the configuration against the project root (`Path(__file__).resolve().parents[2]` of `src/zeod/config.py`, i.e. two levels above `src/zeod/`).
  Source: `config.py:11`, `config.py:123-125`. Verification: no test.

### 3.2 Command-line interface

- **FR-6** The system shall provide the subcommands `label`, `build-dataset`, `train`, `evaluate`, `infer` and
  `pipeline`. `--config` and `--set` are accepted only before the subcommand.
  Source: `src/zeod/cli.py:140-184`. Verification: no test; manual `python -m zeod.cli --help`.
- **FR-7** The system shall seed Python `random` and NumPy with `seed` before running a subcommand.
  Source: `cli.py:23-25`, `cli.py:198`. Verification: no test.
- **FR-8** The system shall log the message and return exit code 1 when loading the config raises
  `FileNotFoundError` or `ValueError`, or when a stage raises `FileNotFoundError`, `ValueError`, `TimeoutError` or
  `RuntimeError`. It shall return 0 on success.
  Source: `cli.py:187-205`. Verification: no test; manual checks listed in section 8.
- **FR-9** The `pipeline` subcommand shall run `label`, `build-dataset`, `train` and `evaluate` in that order,
  with error analysis enabled and the default weights path.
  Source: `cli.py:131-137`. Verification: no test.

### 3.3 Auto-labeling

- **FR-10** The system shall collect the files matching `paths.images_dir/paths.image_glob` in sorted order, keep
  only the first N when `--limit N` is given, and raise `FileNotFoundError` when nothing matches.
  Source: `cli.py:28-33`, `cli.py:40-42`. Verification: no test; manual `label --limit 1` without `images/` -> exit 1.
- **FR-11** The system shall start `llama-server` with `--host`, `--port`, `--ctx-size`, `--n-gpu-layers`,
  `--image-min-tokens` and `--parallel` from the configuration. It shall add `--model` (plus `--mmproj` if set)
  when `model_path` is set, otherwise `-hf <hf_repo>:<hf_quant>`, and finally append `extra_server_args`. It shall
  raise `ValueError` when neither `model_path` nor `hf_repo` is set.
  Source: `src/zeod/labeling/backend.py:79-104`. Verification: no test.
- **FR-12** The system shall raise `RuntimeError` with download and `PATH` instructions when the server binary cannot be launched.
  Source: `backend.py:107-121`. Verification: no test.
- **FR-13** The system shall poll `GET /health` every 2 s until it returns HTTP 200. It shall raise `RuntimeError`
  with the process output if the process exits first, and `TimeoutError` after `startup_timeout_s`.
  Source: `backend.py:125-143`. Verification: no test.
- **FR-14** For each image the system shall send `POST /v1/chat/completions` with the system prompt, a user
  message holding the image as a base64 data URL plus the user prompt, `temperature` and `max_tokens`, and
  return `choices[0].message.content`.
  Source: `backend.py:66-69`, `backend.py:145-174`. Verification: no test.
- **FR-15** On exit from the backend context, and before a restart, the system shall terminate the server
  process, wait up to 10 s and then kill it.
  Source: `backend.py:35-39`, `backend.py:176-190`. Verification: no test.
- **FR-16** The system shall parse a model response by stripping a leading or trailing markdown code fence,
  taking the text from the first `[` to the last `]`, decoding it as JSON and keeping only object items. It shall
  return an empty list for empty, non-array, malformed or truncated input instead of raising.
  Source: `src/zeod/labeling/parser.py:26-52`. Verification: `test_clean_json_array`, `test_empty_array`, `test_empty_string`, `test_whitespace_only`, `test_markdown_code_fence`, `test_prose_before_and_after_json`, `test_malformed_json_returns_empty`, `test_no_array_in_text_returns_empty`, `test_non_list_json_returns_empty`, `test_non_dict_items_are_dropped`, `test_truncated_response_returns_empty`.
- **FR-17** The system shall copy the first present alias of `bbox_2d`, `box` or `box_2d` into `bbox` when `bbox` is missing, and shall not overwrite an existing `bbox`.
  Source: `parser.py:14-23`. Verification: `test_bbox_2d_key_is_normalized_to_bbox`, `test_existing_bbox_key_is_not_overwritten_by_alias`.
- **FR-18** When labeling through `LlamaCppServerBackend`, the system shall map every 4-element numeric bbox from
  the Qwen-VL vision grid to original pixels. The grid size is computed by `smart_resize` with factor 28,
  `min_pixels = image_min_tokens * 28^2` and `max_pixels = (image_max_tokens or 16384) * 28^2`. Detections whose
  bbox cannot be converted, and images that cannot be opened, pass through unchanged.
  Source: `src/zeod/labeling/grid.py:26-71`, `backend.py:192-199`, `config.py:64-75`, `src/zeod/labeling/pipeline.py:19-42`, `cli.py:57`. Verification: `test_smart_resize_dimensions_are_multiples_of_factor`, `test_smart_resize_upscales_small_images_to_meet_min_pixels`, `test_smart_resize_downscales_large_images_to_respect_max_pixels`, `test_smart_resize_roughly_preserves_aspect_ratio`, `test_rescale_bbox_to_pixels_identity_when_grid_matches_original`, `test_rescale_bbox_to_pixels_scales_down_from_larger_grid`, `test_rescale_bbox_to_pixels_handles_non_square_grid` (functions only; the hook inside `label_images` has no test).
- **FR-19** The system shall skip images whose file stem is already in `labels_raw.json` without an `error`
  field, and shall label again those stored with an `error` field.
  Source: `pipeline.py:104-112`. Verification: `test_label_images_is_resumable`, `test_resume_retries_only_previously_failed_images`.
- **FR-20** The system shall store, for every processed image, an entry keyed by file stem with `image_path`,
  `raw_response` and `detections`. When the backend raised an exception, the entry shall get empty detections,
  an empty response and an `error` string, and the run shall continue.
  Source: `pipeline.py:58-75`, `pipeline.py:143-150`. Verification: `test_label_images_survives_backend_exceptions`, `test_full_pipeline_smoke`.
- **FR-21** After 3 consecutive failed requests, the system shall restart the backend and retry the current
  image once. A failed restart shall be logged as a warning and shall not stop the run.
  Source: `pipeline.py:89`, `pipeline.py:125-138`. Verification: `test_label_images_self_heals_after_consecutive_failures`.
- **FR-22** The system shall write `labels_raw.json` (UTF-8, indent 2, `ensure_ascii=False`) every
  `checkpoint_every` processed images and once at the end of the run.
  Source: `pipeline.py:52-55`, `pipeline.py:152-155`. Verification: `test_full_pipeline_smoke` (file exists); checkpoint cadence has no test.

### 3.4 Dataset building

- **FR-23** The system shall skip, and count as skipped, every image with no detections, with an image file
  missing on disk (logged as a warning), or with no YOLO line left after conversion.
  Source: `src/zeod/dataset/build.py:32-78`. Verification: `test_build_dataset_produces_valid_yolo_labels` (`n_skipped == 1`), `test_full_pipeline_smoke` (both cover only the no-detections case; the missing-file and no-line-left cases have no test).
- **FR-24** The system shall drop a detection whose lower-cased, stripped label is not in `class_map` or whose bbox is not a list of 4 numbers.
  Source: `src/zeod/dataset/yolo_convert.py:82-91`. Verification: `test_unknown_label_is_dropped`, `test_malformed_bbox_is_dropped`, `test_label_is_case_and_whitespace_insensitive`, `test_multiple_detections_keep_only_valid_ones`.
- **FR-25** When `dedup_iou_threshold` is not null, the system shall drop each detection whose IoU with an
  earlier kept detection of the same label exceeds the threshold.
  Source: `yolo_convert.py:6-30`, `yolo_convert.py:93-94`. Verification: `test_near_duplicate_same_label_boxes_are_deduped`, `test_near_duplicate_different_label_boxes_are_not_deduped`, `test_dedup_disabled_keeps_both_boxes`.
- **FR-26** The system shall clamp each box to the image bounds and drop it if its width or height is zero or negative after clamping.
  Source: `yolo_convert.py:97-104`. Verification: `test_out_of_bounds_bbox_is_clamped_not_dropped`, `test_degenerate_box_after_clamping_is_dropped`.
- **FR-27** When `head_crop_max_aspect_ratio` is not null, the system shall replace a clamped box whose
  height/width exceeds the ratio with a square of the same width anchored at its top edge.
  Source: `yolo_convert.py:33-54`, `yolo_convert.py:106-109`. Verification: `test_head_crop_squares_a_tall_box_anchored_at_top`, `test_head_crop_leaves_already_square_box_unchanged`, `test_head_crop_disabled_by_default_in_function_signature`.
- **FR-28** The system shall write each remaining box as `<class_id> <cx> <cy> <w> <h>` with values normalised to
  the image size and 6 decimals, keeping it only when both `w` and `h` lie strictly between `min_box_frac` and
  `max_box_frac`.
  Source: `yolo_convert.py:111-119`. Verification: `test_basic_conversion_centers_and_normalizes`, `test_tiny_box_below_min_frac_is_dropped`, `test_huge_box_above_max_frac_is_dropped`.
- **FR-29** The system shall split the valid items by shuffling a copy with `random.Random(seed)` and putting the
  first `round(n * val_split)` items in validation. It shall reject `val_split` outside the open interval (0, 1)
  and leave both the input list and the global RNG state unchanged.
  Source: `src/zeod/dataset/split.py:11-27`. Verification: `test_split_is_deterministic_for_same_seed`, `test_split_proportions_are_approximately_correct`, `test_split_covers_every_item_exactly_once`, `test_different_seeds_can_produce_different_splits`, `test_does_not_mutate_input_list`, `test_does_not_leak_into_global_random_state`, `test_invalid_val_split_raises`.
- **FR-30** The system shall copy each image to `dataset/<split>/images/<stem><ext>`, write its lines to
  `dataset/<split>/labels/<stem>.txt`, and write `dataset/data.yaml` with `path`, `train: train/images`,
  `val: val/images` and `names` ordered by class id.
  Source: `build.py:81-99`, `build.py:123-128`. Verification: `test_build_dataset_produces_valid_yolo_labels`, `test_build_dataset_writes_valid_data_yaml`.
- **FR-31** When no valid item remains, the system shall log an error, write no dataset files and return the statistics.
  Source: `build.py:117-121`. Verification: `test_build_dataset_handles_no_valid_annotations`.
- **FR-32** The `build-dataset` subcommand shall raise `FileNotFoundError` when `labels_raw.json` is missing or
  empty, and shall otherwise print the dataset statistics as JSON.
  Source: `cli.py:61-81`. Verification: no test; manual check without `labels_raw.json` -> exit 1.

### 3.5 Training, evaluation, inference

- **FR-33** The system shall raise `FileNotFoundError` when `dataset/data.yaml` is missing. Otherwise it shall
  fine-tune `YOLO(train.base_weights)` with `epochs`, `imgsz`, `batch`, `patience`, the resolved device,
  `project=runs_dir`, `name=experiment_name`, `exist_ok=True` and `seed`.
  Source: `src/zeod/train.py:14-40`. Verification: no test.
- **FR-34** The system shall use `runs/<experiment_name>/weights/best.pt` as the default weights for `evaluate` and `infer` when `--weights` is not given.
  Source: `train.py:43-45`, `cli.py:96`, `cli.py:120`. Verification: no test.
- **FR-35** The system shall raise `FileNotFoundError` when the weights file is missing. Otherwise it shall run
  `model.val` with `data.yaml`, the resolved device and `evaluate.iou` (no confidence threshold), and print
  `map50`, `map50_95`, `precision` and `recall` as JSON.
  Source: `src/zeod/evaluate.py:24-54`, `cli.py:96-98`. Verification: no test.
- **FR-36** With `--analyze-errors`, the system shall predict every validation image at `evaluate.conf`. It shall
  collect up to `fp_fn_examples` records where the predicted box count exceeds the ground-truth count, and as many
  where it falls short, stop when both lists are full, and print both as JSON.
  Source: `evaluate.py:57-101`, `cli.py:100-113`. Verification: no test.
- **FR-37** The `infer` subcommand shall expand the image glob and raise an error when the weights are missing
  or no image matches. Otherwise it shall predict each image at `--conf` on the `evaluate.device` device and save
  the annotated image under `--output` with the same file name.
  Source: `src/zeod/infer.py:13-39`, `cli.py:116-128`. Verification: no test.
- **FR-38** The system shall resolve device `auto` to CUDA device `0` when `torch.cuda.is_available()`, otherwise
  to `cpu` with a warning, and shall pass any other value through unchanged.
  Source: `src/zeod/device.py:10-24`. Verification: no test.
- **FR-39** The system shall log to stdout in the format `HH:MM:SS [LEVEL] logger: message` at INFO, and shall not add a handler when the root logger already has one.
  Source: `src/zeod/logging_setup.py:9-19`. Verification: no test.

### 3.6 Repository tooling

- **FR-40** `scripts/check_before_push.py` shall extract the `ruff`, `mypy` and `pytest` commands from
  `.github/workflows/*.y*ml`, run the ones whose tool is installed, and exit 1 if any fails. With
  `--install-hook` it shall write `.git/hooks/pre-push`.
  Source: `scripts/check_before_push.py:44`, `85-95`, `98-108`, `156-208`. Verification: no test.
- **FR-41** `scripts/enforce_branch_policy.py` shall report branches other than `release`, `dev` and `test`
  (locally, and on `--remote` when given) and missing required branches, and exit 1 on any violation. With
  `--delete-extra` it shall delete only local branches already merged into an allowed branch.
  Source: `scripts/enforce_branch_policy.py:19`, `40-100`. Verification: no test.
- **FR-42** `scripts/build_release_artifact.py` shall rebuild `release/` with a PyInstaller one-file executable
  of `src/zeod/cli.py` (unless `--skip-exe`), a zip of the executable with `README.md`, `LICENSE` and
  `CHANGELOG.md`, and a `.sha256` file per artefact.
  Source: `scripts/build_release_artifact.py:39-130`. Verification: no test (CI `release.yml` runs it on tags).
- **FR-43** `scripts/collect_quality_metrics.py` shall report test pass rate, coverage, radon complexity and
  maintainability with ruff findings, and declared Python versions and CI operating systems. It shall write the
  report as JSON with `--output` and exit 1 when the mean score is below `--fail-under`.
  Source: `scripts/collect_quality_metrics.py:53-210`. Verification: no test.
- **FR-44** `scripts/install_dependencies.py` shall install the package (`--dev`: with dev extras), regenerate
  `requirements-dev.lock` in a clean virtual environment with `--lock`, and install from it with `--from-lock`.
  Source: `scripts/install_dependencies.py:30-98`. Verification: no test.

### 3.7 Declared but not implemented

| Claim | Where claimed | What the code does |
| --- | --- | --- |
| Device selection picks "CUDA, MPS or CPU" | `docs/architecture.md`, `docs/quality-iso25010.md` sections 3 and 8 | `auto` resolves only to CUDA 0 or CPU (`device.py:10-24`) |
| Switching to MiniCPM-V or Moondream2 needs only `hf_repo` and the prompt; "the rest of the pipeline ... is model-agnostic" | `README.md`, model-choice section | `LlamaCppServerBackend.bbox_rescaler` always applies the Qwen grid rescale (`backend.py:192-199`, `cli.py:57`). Inference: another model's coordinates would be rescaled as if they were Qwen-grid values |
| Self-consistency labeling (2-3 passes, vote) and two-stage person-then-head grounding | `README.md`, "Bottom line"; notebook cell "Ideas to improve metrics" | Single pass only (`pipeline.py:78-164`) |
| `.github/workflows/branch-policy.yml`, weekly run | `docs/branching.md`, "Enforcement" | No such file; a `branch-policy` job in `python-ci.yml:56-66`, no schedule |
| Scheduled secret scan; ruff `flake8-bandit` rules | `docs/quality-iso25010.md` section 6 | gitleaks runs on push and PR (`python-ci.yml:45-54`); ruff selects `E, F, I, UP, B` (`pyproject.toml:48`) |
| gitleaks and `detect-private-key` in pre-commit; ruff `S`; `requirements-dev.lock`; `ci.yml` | `SECURITY.md` | `.pre-commit-config.yaml` has ruff, ruff-format, trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files only; no lock file; workflow is `python-ci.yml` |
| Coverage `fail_under` 85; ruff `C90` complexity limit 10 | `CONTRIBUTING.md` | Neither is configured in `pyproject.toml` |
| Profiling and screenshot capture of the application | `scripts/profile_application.py`, `scripts/capture_usage_screenshots.py` | Both target `quality_template`, which is absent; the profiler fails on import (checked) |

## 4. Interfaces

### 4.1 Command line (`zeod` = `python -m zeod.cli`, `pyproject.toml:34`)

| Element | Arguments | Default | Source |
| --- | --- | --- | --- |
| global | `--config PATH` | `configs/default.yaml` | `cli.py:148` |
| global | `--set KEY=VALUE` (repeatable) | none | `cli.py:149-155` |
| `label` | `--limit N` | all images (also when `N` is 0) | `cli.py:158-160`, `cli.py:41` |
| `build-dataset` | none | - | `cli.py:162-163` |
| `train` | none | - | `cli.py:165-166` |
| `evaluate` | `--weights PATH`, `--analyze-errors` | `runs/<name>/weights/best.pt`, off | `cli.py:168-171` |
| `infer` | `IMAGES` (glob), `--weights`, `--output DIR`, `--conf FLOAT` | -, best.pt, `runs/infer_output`, `0.25` | `cli.py:173-178` |
| `pipeline` | `--limit N` | all images | `cli.py:180-182` |
| exit codes | `0` success, `1` handled error, `2` argparse usage error | - | `cli.py:187-205` |

### 4.2 HTTP client calls to `llama-server`

| Method, path | Request | Response used | Timeout | Source |
| --- | --- | --- | --- | --- |
| `GET /health` | none | status 200 = ready | 5 s per poll, 2 s between polls, `startup_timeout_s` total (240) | `backend.py:125-143` |
| `POST /v1/chat/completions` | `{"messages": [{"role": "system", "content": <system_prompt>}, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:<mime>;base64,..."}}, {"type": "text", "text": <user_prompt>}]}], "temperature": t, "max_tokens": n}` | `choices[0].message.content`; non-2xx raises | `request_timeout_s` (120) | `backend.py:145-174` |

MIME types: `.png`, `.jpg`/`.jpeg`, `.bmp`; anything else is sent as `application/octet-stream` (`backend.py:29`, `backend.py:66-69`).

### 4.3 Server command line built by the client

`<server_binary> --host H --port P --ctx-size C --n-gpu-layers G --image-min-tokens M --parallel N` followed
by `--model F [--mmproj F]` or `-hf REPO:QUANT`, then `extra_server_args` (`backend.py:81-104`).

### 4.4 Files

| File | Producer -> consumer | Format | Source |
| --- | --- | --- | --- |
| `configs/default.yaml` | operator -> all stages | YAML, keys in section 5 | `config.py` |
| `labels_raw.json` | `label` -> `build-dataset`, notebook | JSON object, see 5.1 | `pipeline.py:45-55` |
| `dataset/{train,val}/images/*` | `build-dataset` -> `train`, `evaluate` | copies of the source images | `build.py:81-88` |
| `dataset/{train,val}/labels/<stem>.txt` | `build-dataset` -> Ultralytics | one `<id> <cx> <cy> <w> <h>` line per box, `\n`-separated, no trailing newline | `build.py:89`, `yolo_convert.py:119` |
| `dataset/data.yaml` | `build-dataset` -> `train`, `evaluate` | `path: <abs dir>`, `train: train/images`, `val: val/images`, `names: {id: name}` | `build.py:92-99` |
| `runs/<experiment_name>/` | Ultralytics `train` -> `evaluate`, `infer`, notebook | Ultralytics run directory; `weights/best.pt` | `train.py:27-45` |
| `runs/infer_output/<name>` | `infer` | annotated images | `infer.py:33-35`, `cli.py:176` |
| stdout | `build-dataset`, `evaluate` | JSON: dataset stats; metrics; `false_positive_examples` / `false_negative_examples` | `cli.py:81`, `cli.py:98`, `cli.py:112-113` |

## 5. Data

### 5.1 Entities

| Entity | Fields | Source |
| --- | --- | --- |
| Label entry (`labels_raw.json`, keyed by image file stem) | `image_path` (string; absolute, because image paths are resolved against the project root), `raw_response` (string), `detections` (list), optional `error` (string) | `pipeline.py:143-150`, `cli.py:29-30`, `config.py:123-125` |
| Detection | `label` (string), `bbox` `[x_min, y_min, x_max, y_max]` in original-image pixels after rescaling; other keys from the model are kept | `parser.py:17-23`, `pipeline.py:19-42` |
| YOLO line | `class_id cx cy w h`, normalised, 6 decimals | `yolo_convert.py:111-119` |
| `DatasetStats` | `n_total_labeled`, `n_valid`, `n_skipped`, `n_train`, `n_val`, `class_counts` (count per class id) | `build.py:19-26` |
| `ValMetrics` | `map50`, `map50_95`, `precision`, `recall` | `evaluate.py:16-21` |
| FP/FN record | `name`, `image_path`, `n_pred`, `n_gt` | `evaluate.py:92` |
| Configuration | `seed`, `paths`, `labeling`, `llama_cpp`, `dataset`, `train`, `evaluate` (defaults in `docs/README.md`, "Configuration") | `config.py:15-121`, `configs/default.yaml` |

### 5.2 Storage

All data lives in local files under the project root. The inputs and generated artefacts are git-ignored:
`/images/`, `/labels_raw.json`, `/dataset/`, `/runs/`, `*.pt`, `*.gguf`, `*.onnx` (`.gitignore`). The source photos
are not in the repository. No database is used.

## 6. Non-functional requirements

- **NFR-1** Server start-up shall be bounded by `startup_timeout_s` (240 s default), with `/health` polled every 2 s at a 5 s timeout per poll. Source: `configs/default.yaml:56`, `backend.py:125-143`.
- **NFR-2** Each labeling request shall time out after `request_timeout_s` (120 s default). Source: `default.yaml:57`, `backend.py:167-171`.
- **NFR-3** Server shutdown shall take at most about 10 s before the process is killed. Source: `backend.py:180-184`.
- **NFR-4** A failure on one image shall not stop the labeling run; the backend shall be restarted after 3 consecutive failures. Source: `pipeline.py:74-75`, `pipeline.py:125-138`.
- **NFR-5** At most `checkpoint_every` (20) images of labeling work shall be lost on interruption. Source: `default.yaml:34`, `pipeline.py:152-153`.
- **NFR-6** The seeded parts of a run shall be reproducible for a given `seed` (42): Python and NumPy RNG, train/val split, Ultralytics `seed`. The labeling request sends no seed (only `temperature`, 0.1), so VLM output is not covered. Source: `cli.py:23-25`, `split.py:22`, `train.py:37`, `backend.py:153-166`.
- **NFR-7** The server shall bind to `127.0.0.1` by default and use one slot (`--parallel 1`); requests are sent one at a time. Source: `default.yaml:42`, `config.py:42-48`, `pipeline.py:120-121`.
- **NFR-8** The package shall run on Python 3.10 to 3.13 (declared). Only 3.12 is exercised by CI. Source: `pyproject.toml:10`, `python-ci.yml:27`.
- **NFR-9** Unit tests shall run without a GPU, a network or a model: they use `MockBackend` (plus a stub `VLMBackend` subclass in the restart test), synthetic images and `tmp_path`, and import no Ultralytics. Source: `tests/test_pipeline_smoke.py`, `backend.py:202-223`.
- **NFR-10** Code shall pass `ruff check` (rules `E, F, I, UP, B`, line length 115, target py310; excluding `legacy`, `scratch`, `notebooks`) and `ruff format --check`. Source: `pyproject.toml:42-48`, `python-ci.yml:36-40`.
- **NFR-11** gitleaks shall run on every push and PR to `release`, `dev` and `test`, on a checkout with full history (`fetch-depth: 0`). Which commits it scans for a given event is decided by `gitleaks/gitleaks-action@v2`, not by this repository. Source: `python-ci.yml:3-8`, `python-ci.yml:45-54`.
- **NFR-12** CI shall run with `contents: read` and cancel superseded runs on the same ref. Source: `python-ci.yml:10-15`.
- **NFR-13** Every release artefact shall have a SHA-256 checksum file next to it. Source: `build_release_artifact.py:122-125`.
- **NFR-14** Files larger than 5000 KB shall be refused by the pre-commit hook. Source: `.pre-commit-config.yaml` (`check-added-large-files --maxkb=5000`).
- **NFR-15** Config, label and dataset text files shall be read and written as UTF-8. Source: `config.py:158`, `pipeline.py:47-55`, `build.py:89`, `build.py:98`.
- **NFR-16** IoU deduplication compares each detection with every kept one, so its cost is quadratic in the detection count. `README.md` reports 3.60 ms for 128 distinct boxes (author's figure, not reproduced here). Source: `yolo_convert.py:19-30`, `benchmarks/test_labeling_performance.py`.

## 7. Constraints and dependencies

There is no lock file on `dev`; versions are ranges from `pyproject.toml`.

| Dependency | Constraint | Group | Source |
| --- | --- | --- | --- |
| ultralytics | `>=8.3.0,<9` | runtime | `pyproject.toml:14` |
| pydantic | `>=2.7,<3` | runtime | `pyproject.toml:15` |
| pyyaml | `>=6.0,<7` | runtime | `pyproject.toml:16` |
| requests | `>=2.31,<3` | runtime | `pyproject.toml:17` |
| numpy | `>=1.26,<3` | runtime | `pyproject.toml:18` |
| pillow | `>=10.0,<12` | runtime | `pyproject.toml:19` |
| tqdm | `>=4.66,<5` | runtime | `pyproject.toml:20` |
| torch | not declared; imported by `device.py`; CI installs the CPU wheel; README pins `2.6.0+cu124` / torchvision `0.21.0+cu124` for CUDA | runtime (indirect) | `device.py:19`, `python-ci.yml:31`, `README.md` |
| pytest `>=8.0,<9`, pytest-cov `>=5.0,<6`, ruff `>=0.6,<1`, radon `>=6.0`, pytest-benchmark `>=4.0` | | `dev` | `pyproject.toml:24-30` |
| pyinstaller | `>=6.0` | `release` | `pyproject.toml:31` |
| hatchling | build backend | build | `pyproject.toml:1-3` |
| llama.cpp `llama-server` | version not pinned; must support `--image-min-tokens`, `--parallel`, `-hf` | external binary | `backend.py:81-104` |
| Model | `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF`, quant `Q4_K_M` | default config | `default.yaml:46-47` |
| Base detector | `yolov8n.pt` | default config | `default.yaml:72` |
| CI actions | `actions/checkout@v4`, `actions/setup-python@v5`, `gitleaks/gitleaks-action@v2`, `actions/upload-artifact@v4`, `actions/download-artifact@v4`, `softprops/action-gh-release@v2` | CI | `.github/workflows/` |
| pre-commit hooks | `ruff-pre-commit` `v0.6.9`, `pre-commit-hooks` `v4.6.0` | dev | `.pre-commit-config.yaml` |

Licence: MIT (`LICENSE`, `pyproject.toml`).

## 8. Acceptance

Automated:

1. `pytest`: 60 tests in `tests/`, all must pass. Checked on 2026-11-18 with `PYTHONPATH=src` and Python 3.13.3:
   60 passed. These tests cover FR-1 to FR-4, FR-16 to FR-31 and NFR-9.
2. `ruff check .` and `ruff format --check .` must pass (NFR-10; CI `python-ci.yml:36-40`). Not run here.
3. `pytest benchmarks --benchmark-only` must complete. The README timings serve as a reference, not a threshold.
   Not run here.
4. CI `python-ci.yml` must be green on `dev`, `test` and `release`. CI state was not checked here.

Manual checks (for requirements without tests):

| Check | Expected | Covers | Done here |
| --- | --- | --- | --- |
| `python -m zeod.cli --help` | six subcommands listed | FR-6 | yes |
| `python -m zeod.cli --config nonexistent.yaml build-dataset` | error logged, exit 1 | FR-1, FR-8 | yes |
| `python -m zeod.cli --set badoverride build-dataset` | error logged, exit 1 | FR-3, FR-8 | yes |
| `python -m zeod.cli build-dataset` without `labels_raw.json` | "No labels found ... run `zeod label` first", exit 1 | FR-32 | yes |
| `python -m zeod.cli label --limit 1` without `images/` | "No images matching ...", exit 1 | FR-10 | yes |
| `python -m zeod.cli pipeline --limit 15` with `llama-server` and a GPU | all four stages complete; metrics JSON printed | FR-9 to FR-15, FR-18, FR-33 to FR-36 | no (README reports it done on an RTX 3060 Ti) |
| `python -m zeod.cli infer "images/*.png"` after training | annotated copies in `runs/infer_output/` | FR-37 | no |
| Notebook `visualize_annotations` on a labeled sample | boxes align with heads | FR-18 (rescale correctness) | no |

## 9. Open questions

1. Who uses the trained detector, and where does it run? The code ends at `infer` writing annotated images; no
   deployment target is stated.
2. What metric is acceptable? No mAP, precision or recall threshold appears in code, tests or config
   (`docs/quality-iso25010.md` section 1 says the same).
3. Which llama.cpp release was used? The flags `--image-min-tokens`, `--parallel` and `-hf` are required, but no version is pinned.
4. Where does the input dataset come from, under what licence, and what rules apply to handling photos of
   identifiable people? `docs/task.txt` mentions a 251 MB download. The repository holds no images and states no
   data-handling policy beyond running locally.
5. Is `dev` still the line of development? `origin/release` and `origin/test` each carry 22 later commits that are
   not on `dev`; the commit tagged `v1.1.0` (2026-11-08) is on `release`.
6. What is the version? On `dev`, `pyproject.toml` says `0.2.0`, `CHANGELOG.md` says `1.0.0`, and tag `v1.1.0` exists.
   Partly answered: `pyproject.toml` on `origin/release` says `1.1.0` (`git show origin/release:pyproject.toml`).
   Which number `dev` should carry is still open.
7. Was the released executable ever run? The build bundles no `configs/default.yaml`, and the default config path
   is derived from `__file__` (`build_release_artifact.py:58-76`, `config.py:11-12`). Inference: it likely needs
   `--config`.
8. What caused the server hang after about 100 requests? The code comment blames slot scheduling
   (`config.py:42-47`). The undrained stdout pipe (`backend.py:108-114`, read only at `backend.py:131`) is an
   alternative explanation that was not tested.
9. Should `train.device: 0` work as the YAML comment says? Pydantic rejects the integer (checked with
   `--set train.device=0`).
10. Should MPS be supported, as `docs/architecture.md` states?
11. Should `build-dataset` remove a previous `dataset/` before writing? Leftover files from earlier splits are
    possible (`build.py:81-89`).
12. Should the notebook's label preview apply the same head crop as `build-dataset`? It calls
    `build_valid_items` without `head_crop_max_aspect_ratio`, which defaults to `None` (`build.py:38`).
13. Is `legacy/agent_scaffolding_cpp_qt/` still needed? `README.md` calls it "safe to remove entirely".
