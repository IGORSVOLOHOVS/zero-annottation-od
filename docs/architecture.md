# Architecture

The whole project is one idea: **replace the human labeller with a
vision-language model, then throw the VLM away.** A large model looks at raw
photos once and writes boxes; those boxes become a YOLO dataset; a small
detector is trained on it and is what actually ships. The expensive model is a
build-time tool, not a runtime dependency.

That is why the code is shaped as a pipeline of stages that hand files to each
other rather than as a library of objects. Each stage can be run alone, its
output inspected on disk, and re-run without repeating the ones before it —
which matters when the first stage takes hours on a GPU.

## The stages

```
src/zeod/
  cli.py              one entry point, one subcommand per stage
  config.py           AppConfig - every path and threshold in one place
  labeling/           stage 1: raw photos -> labels_raw.json
    backend.py            talks to the local VLM
    grid.py               smart_resize, the patch grid the VLM expects
    parser.py             turns free-form model text into detections
    pipeline.py           walks the image folder, calls the backend
  dataset/            stage 2: labels_raw.json -> a YOLO dataset
    yolo_convert.py       deduplication, box repair, YOLO format
    split.py              train/val split
    build.py              writes the directory layout YOLO wants
  train.py            stage 3: fine-tune YOLOv8
  evaluate.py         stage 4: metrics, and mining false positives
  infer.py            stage 5: run the trained detector
  device.py           where the work runs - CUDA, MPS or CPU
  logging_setup.py    one logging configuration for every stage
```

`zeod pipeline` runs label, build-dataset, train and evaluate in order;
`zeod infer` is separate, because it is what you run afterwards rather than
part of producing the detector.

## The dependency rule

Stages depend on the stage before them only through **files on disk**, never
through imports. `dataset/build.py` imports nothing from `labeling/`; it reads
`labels_raw.json`. That is the whole reason a labelling run costing hours can be
reused while the dataset conversion is rewritten a dozen times.

`config.py` is the one thing every stage imports. Paths and thresholds live
there rather than being passed down through five call layers, so changing where
the raw photos live is one edit.

## Where the judgement lives

Two functions carry the decisions that determine dataset quality, and both are
in `dataset/yolo_convert.py`:

**`_dedup_by_iou`** — a VLM will happily emit two near-identical boxes for one
person. Left alone, YOLO learns from two "ground truths" for one head. Later
boxes overlapping an earlier one of the same label by more than the IoU
threshold are dropped.

**`_crop_tall_box_to_head`** — asked for heads, this VLM draws tight
roughly-square boxes for `helmet` (median height/width ≈ 0.9) but falls back to
full-body boxes for `no_helmet` (median ≈ 2.5), whatever the prompt says. A box
taller than the aspect threshold is squared using its own width and anchored at
the top edge, because the head is the topmost part of a person in every pose in
this dataset. It is a proxy, not a measurement — a shoulder-width box yields
head-and-shoulders — and it is the single most consequential guess in the
project, which is why it is one named function with the reasoning written down
rather than three lines inside a loop.

`parser.py` is deliberately forgiving: markdown fences, prose around the JSON,
truncated output and empty responses all return `[]` rather than raising. One
bad response should cost one image, not the run.

## Known limits

`label_images` is the most complex function in the codebase (cyclomatic
complexity 16) — it carries retry handling, progress reporting and incremental
resume in one place. It is the obvious next refactor.

Deduplication compares each detection against every one already kept, so its
cost grows with the square of the detection count. `benchmarks/` measures the
shape: harmless at four boxes per photo, 3.6 ms at 128. See
`docs/quality-iso25010.md`.
