# Quality assessment — ISO/IEC 25010

ISO/IEC 25010 defines eight product quality characteristics. Four are
**measured** by `scripts/collect_quality_metrics.py`; the other four need
judgement and are argued below, citing those measurements rather than asserting
quality.

```bash
python scripts/collect_quality_metrics.py --output quality-report.json
```

Latest run, 2026-08-05 — overall measured score **84.5**:

| Characteristic | Score | Evidence |
| --- | --- | --- |
| Functional suitability | 100.0 | 60 tests, 0 failures |
| Reliability | 88.3 | 88.3 % branch coverage |
| Maintainability | 74.5 | mean cyclomatic complexity 2.79, maintainability index 74.5, 0 lint findings |
| Portability | 75.0 | two operating systems, one interpreter version |

---

## 1. Functional suitability — measured

*Does it do what it claims, correctly and completely?*

The claim is unusually checkable: produce a helmet / no-helmet detector without
a human labelling anything. 60 tests, all passing, and they test the parts where
the claim can quietly fail — `test_parser.py` on malformed model output,
`test_yolo_convert.py` on deduplication and box repair, `test_split.py` on the
train/val split, `test_dataset_validity.py` on the directory YOLO is handed.

**Known gap:** nothing asserts a detection-quality floor. The pipeline is tested
to produce a *well-formed* dataset, not a *good* one; whether the resulting
detector is accurate is measured by `zeod evaluate` and read by a person.
That is defensible — accuracy depends on the photos supplied — but it means the
green tick covers the plumbing, not the outcome.

## 2. Performance efficiency — assessed, with measurements

*Time behaviour, resource use, capacity.*

Wall-clock time is dominated by the VLM and by YOLO training, both of which
depend on the machine's GPU rather than on this code.
`benchmarks/test_labeling_performance.py` therefore measures the per-image work
*around* the model. Measured on this machine:

| Operation | Median |
| --- | --- |
| `smart_resize` | 0.5 µs |
| Parse a junk response | 1.8 µs |
| Deduplicate 4 detections | 3.9 µs |
| Parse 16 detections | 30.9 µs |
| Parse 16 detections inside a markdown fence | 31.9 µs |
| Deduplicate 128 identical detections | 82.5 µs |
| Deduplicate 32 distinct detections | 228.9 µs |
| Train/val split of 10,000 items | 1.70 ms |
| Deduplicate 128 distinct detections | 3.62 ms |

The shape worth knowing is `_dedup_by_iou`. It compares each detection against
every one already kept, so cost grows with the square of the count: 4 → 32
detections is 8× the input for **58×** the time; 32 → 128 is 4× the input for
**16×** the time. At four boxes per photo that is 4 µs and irrelevant. On a
crowded site at 128 boxes it is 3.6 ms per image — still small against a VLM
call, but no longer free, and it would become the bottleneck if the model were
ever replaced by something fast.

The collapsing case is 44× cheaper than the distinct case at the same count
(82.5 µs against 3.62 ms), because a box that matches an early neighbour exits
the comparison immediately. Real photos sit between the two.

Stripping a markdown fence costs 1 µs on top of a 31 µs parse — 3 %. Being
forgiving about model output is close to free.

## 3. Compatibility — assessed

*Co-existence and interoperability.*

Every stage hands the next one an ordinary file: `labels_raw.json`, then a YOLO
directory in the layout Ultralytics already expects. Nothing here invents a
format, so the dataset can be fed to any YOLO trainer and the labels can be
inspected in any JSON viewer.

`device.py` picks CUDA, MPS or CPU at runtime, so the same commands run on a
workstation with a GPU and on a laptop without one — slowly, but they run.

**Known limit:** the labelling backend targets one local VLM family and its
patch-grid arithmetic (`grid.py`) is written to that family's expectations.
Another model would need a new backend, though the parser is deliberately
model-agnostic.

## 4. Usability — assessed

*Can someone who did not write it get a result?*

One command per stage, one command for all of them, and stages that resume:
because each reads the previous stage's file from disk, a failed training run
does not cost the hours of labelling that preceded it. That is the usability
decision that matters most in a pipeline with an expensive first step.

The README leads with what the tool is for and what it produces, and
`docs/architecture.md` explains the two functions that decide dataset quality,
so a reader knows where to look before changing anything.

**Known gap:** no GUI, and the error mining in `zeod evaluate` produces images a
person still has to look at. The audience is assumed comfortable with a
terminal.

## 5. Reliability — measured

*Does it keep working?*

88.3 % branch coverage across 60 tests, concentrated on the parsing and
conversion code — which is correct, because that is where untrusted input
arrives. A VLM's reply is free-form text: `parse_json_response` returns `[]` for
fences, prose, truncation and empty strings rather than raising, so one bad
response costs one image instead of the run.

**Known limit:** the uncovered branches are mostly in `train.py` and
`evaluate.py`, which need real weights and a GPU to exercise. They are the
least-tested and longest-running parts of the system.

## 6. Security — assessed

*Confidentiality, integrity, resistance to misuse.*

Ruff's `flake8-bandit` rules run over the tree on every push; current
outstanding findings: **0**. A scheduled `secret-scan` workflow searches the
whole git history for credentials, not just the working tree.

The stronger property is architectural: **the pipeline is entirely local.** The
VLM runs on the machine, so construction-site photographs — which show
identifiable people, often without their consent to any cloud service — are
never uploaded anywhere. That was a design choice, not an accident of
implementation.

**Known limit:** model weights are trusted implicitly, and `train.py` executes
whatever Ultralytics does with the configuration it is given.

## 7. Maintainability — measured

*Can it be changed safely?*

Mean cyclomatic complexity 2.79, maintainability index 74.5, zero outstanding
lint findings.

The outlier is `label_images` at complexity 16 — retry handling, progress
reporting and incremental resume in one function. It is the obvious next
refactor, and the fact that it is also the slowest-running code makes it the
least pleasant to debug.

Stages depend on each other only through files, never imports (verified: nothing
in `dataset/` imports from `labeling/`), so a stage can be rewritten without
reading the others.

## 8. Portability — measured

*Where does it run?*

Score 75.0: the release workflow builds on **ubuntu-latest and windows-latest**
and publishes both artefacts, while CI tests on **Python 3.12 only**. The README
badge claims 3.10–3.13, so three of those four versions are claimed and
untested.

`device.py` handles CUDA, MPS and CPU, so the code paths for all three exist —
but CI has no GPU, so only the CPU path is exercised anywhere automatic.

**To raise this honestly:** add a Python 3.10 cell — the oldest claimed version
is where a syntax or typing feature would break first.
