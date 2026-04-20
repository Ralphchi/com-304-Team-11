# COM-304 Team 11 — Extension: Alternate Masking Strategies for nano4M

Extension project exploring whether structured masking (block on images, span
on text, mixed) outperforms the uniform-random baseline used by nano4M.

See `COM_304_Extension_plan___Team_11.pdf` (separate document) for the full
proposal.

---

## Team roles

| Person  | Role                                                                    | Branch                              |
|---------|-------------------------------------------------------------------------|-------------------------------------|
| Nacer   | V1 Block masking on image modalities                                    | `extension/nacer-block-masking`     |
| Ricardo | V2 Span masking on scene descriptions                                   | `extension/ricardo-span-masking`    |
| Gabriel | V3 Mixed masking + controlled baseline retraining                       | `extension/gabriel-mixed-masking`   |
| Ralph   | Scene-description parser, Hungarian matcher, per-modality metric harness | `extension/ralph-evaluation`        |

---

## Repository layout (relevant bits)

```
nano4M/
├── nanofm/
│   ├── data/multimodal/masking.py       # <-- teammates add BlockMasking / SpanMasking / MixedMasking here
│   └── evaluation/                      # <-- Ralph's scaffold
│       ├── scene_parser.py              #     parse CLEVR scene_desc -> SceneObject list
│       ├── hungarian_match.py           #     align predicted objects to GT by position
│       ├── metrics.py                   #     depth / normals / RGB / scene_desc metrics
│       └── eval_harness.py              #     load checkpoint -> eval 500 samples -> metrics
├── tests/
│   ├── test_scene_parser.py             #     Week-1 gate
│   └── test_metrics.py
├── cfgs/nano4M/
│   ├── multiclevr_d6-6w512.yaml             (reference)
│   ├── multiclevr_d6-6w512_baseline.yaml    (Gabriel)
│   ├── multiclevr_d6-6w512_block.yaml       (Nacer)
│   ├── multiclevr_d6-6w512_span.yaml        (Ricardo)
│   └── multiclevr_d6-6w512_mixed.yaml       (Gabriel)
├── run_training.py
└── run_evaluation.py                       # <-- Ralph's harness entry point
```

---

## Workflow

### 1. First-time setup

```bash
git clone https://github.com/Ralphchi/com-304-Team-11.git
cd com-304-Team-11
git remote add upstream https://github.com/EPFL-VILAB/com-304-FM-project-2026.git
git checkout extension/<your-branch>
```

Pull updates from professor's repo periodically:

```bash
git fetch upstream
git merge upstream/main           # resolve conflicts on your branch
```

### 2. Day-to-day

- **Always work on your own extension branch**, never commit directly to `main`.
- Pull from `origin/main` before starting each session:
  ```bash
  git checkout main && git pull && git checkout extension/<yours> && git merge main
  ```
- Commit frequently with descriptive messages.
- Push to your branch on `origin`:
  ```bash
  git push origin extension/<yours>
  ```

### 3. Opening a PR

When your slice is ready for review:

1. `git push origin extension/<yours>`
2. Open a Pull Request on GitHub: `extension/<yours>` → `main`.
3. Request review from at least one teammate.
4. After approval + green tests, squash-merge into `main`.
5. Everyone pulls the new `main` and rebases/merges into their extension branch.

---

## Weekly gates (from proposal Section V)

### Week 1 — Implementation + validation (~3 GPU-h each)
- Nacer, Ricardo, Gabriel implement their masking variants in `nanofm/data/multimodal/masking.py`.
- Ralph builds the parser + metric harness (`nanofm/evaluation/`) with unit tests.
- **End-of-week gate**:
  - Every variant produces visually-valid masks on a held-out batch:
    - Image modalities: visual inspection of the mask overlay
    - scene_desc: print the masked span positions
  - Parser unit tests pass: `cd nano4M && python -m pytest tests/ -v`

### Week 2 — Training (~4 GPU-h each)
- Queue all four runs on SCITAS by end of Week 1 to absorb scheduling delay:
  ```bash
  # From a compute node, one run per variant:
  OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 run_training.py \
      --config cfgs/nano4M/multiclevr_d6-6w512_baseline.yaml
  OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 run_training.py \
      --config cfgs/nano4M/multiclevr_d6-6w512_block.yaml
  OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 run_training.py \
      --config cfgs/nano4M/multiclevr_d6-6w512_span.yaml
  OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 run_training.py \
      --config cfgs/nano4M/multiclevr_d6-6w512_mixed.yaml
  ```
- Ralph finalises the metric harness on the Notebook 4 checkpoint while
  the runs complete.
- Monitor validation loss daily; extend budget on any variant that is clearly
  underfitting (proposal Section VI).

### Week 3 — Evaluation (~2 GPU-h)
- Run evaluation across all four checkpoints on 500 held-out samples × 3 seeds:
  ```bash
  for v in baseline block span mixed; do
    for seed in 0 1 2; do
      python run_evaluation.py \
        --ckpt outputs/nano4M/multiclevr_d6-6w512_${v}/checkpoint-final.safetensors \
        --config cfgs/nano4M/multiclevr_d6-6w512_${v}.yaml \
        --num-samples 500 --seed ${seed} \
        --output results/${v}_seed${seed}.json
    done
  done
  ```
- Block-size and span-length ablations if time permits.
- Start drafting the 4-page report.

### Week 4 — Report & deliverables (~1 GPU-h)
- Final 4-page report, website, figures, presentation. No new training.

---

## Metrics (per modality)

| Modality    | Primary metric(s)               | Secondary          |
|-------------|---------------------------------|--------------------|
| Depth       | AbsRel, RMSE, δ₁                |                    |
| Normals     | Mean angular error (degrees)    |                    |
| RGB         | SSIM vs Cosmos reconstructions  | FID (noted unreliable on CLEVR) |
| Scene desc  | Per-field accuracy (position ±3px, shape/color/material exact), exact-sequence match | |

Reported as mean ± std across 3 generation seeds per variant.

---

## Key implementation pointers

### Adding a new masking class (Nacer / Ricardo / Gabriel)

The current factory `nanofm.data.multimodal.create_multimodal_masked_dataloader`
hard-codes `SimpleMultimodalMasking`. You will need to:

1. Add your new `BlockMasking` / `SpanMasking` / `MixedMasking` class in
   `nano4M/nanofm/data/multimodal/masking.py`, exposing the same `__call__`
   interface as `SimpleMultimodalMasking`.
2. Extend `create_multimodal_masked_dataloader` to accept an optional
   `masking` dict (e.g. with a `_target_` Hydra-style key) and use it to
   construct the masking transform. Fall back to `SimpleMultimodalMasking`
   when not provided (keeps the baseline working unchanged).
3. Wire the `masking:` section in your variant's YAML config (placeholder
   already present in the config file).

Week-1 gate: run
  ```
  python -m nanofm.data.multimodal.masking  # or inline script
  ```
to visualise the mask on a sample batch for every variant — no silent bugs.

### Adding the metrics code (Ralph)

Already scaffolded in `nano4M/nanofm/evaluation/` with unit tests. To fill in
`EvalHarness.load` and `.run`, follow the TODOs in `eval_harness.py`. The
generation loop should call `FourM.generate_one_modality_roar` per target
modality with the other modalities as context.

---

## Useful one-liners

```bash
# Sanity check: every module imports cleanly
python -c "from nanofm.evaluation import parse_scene_description; print('OK')"

# Run parser tests
cd nano4M && python -m pytest tests/test_scene_parser.py -v

# Check which of your files are staged
git status -sb

# See what diverges on your branch from main
git log --oneline main..HEAD

# Pull upstream (professor) updates into main
git checkout main && git fetch upstream && git merge upstream/main && git push origin main
```

---

## Contact / points of pain

- **SCITAS queue latency**: queue Week-2 jobs early; proposal allocates buffer.
- **Silent masking bugs**: Week-1 gate enforces visual mask inspection for every variant.
- **Training instability**: monitor val loss daily in Week 2; extend budget if a variant underfits.
