# COM-304 Team 11 — Alternate Masking Strategies for nano4M

Extension comparing four masking strategies on CLEVR at the same compute
budget: V0 random baseline, V1 block masking on images, V2 context-block
masking on images, V3 span masking on scene descriptions. See
`COM_304_Extension_plan_Team_11.pdf` for the proposal.

---

## Team

| Person  | Lane                                                     |
|---------|----------------------------------------------------------|
| Gabriel | V0 baseline + V2 context-block masking + `validate_masks.py` |
| Nacer   | V1 block masking on image modalities                     |
| Ricardo | V3 span masking on scene_desc + μ calibration             |
| Ralph   | Scene-desc parser, Hungarian matching, metrics, statistical tests, Cosmos sanity, `inspect_scene_desc.py`, eval harness |

---

## Workflow

Single-branch: everyone commits and pushes directly to `main`. We tried
per-teammate branches in Week 1 and the branch admin became a bigger time
sink than the actual code, so we collapsed to `main` only on 2026-04-28.

Day-to-day:

```bash
git pull --rebase origin main      # before starting
# ... edit code ...
cd nano4M && python -m pytest tests/ -q
cd nano4M && python scripts/validate_masks.py   # 4/4 PASS expected
git add <files>
git commit -m "<descriptive message>"
git push origin main
```

Sync the professor's upstream when needed:

```bash
git fetch upstream && git merge upstream/main && git push origin main
```

---

## Repository layout

```
nano4M/
├── nanofm/
│   ├── data/multimodal/
│   │   ├── masking.py                  # SimpleMultimodalMasking (V0 parent)
│   │   ├── block_masking.py            # BlockMasking (V1, Nacer)
│   │   ├── context_block_masking.py    # ContextBlockMasking (V2, Gabriel)
│   │   └── span_masking.py             # SpanMasking (V3, Ricardo)
│   └── evaluation/
│       ├── scene_parser.py             # parse CLEVR scene_desc → SceneObject
│       ├── hungarian_match.py          # align predicted to GT objects
│       ├── metrics.py                  # depth / normals / RGB / scene_desc
│       ├── statistical_tests.py        # paired Wilcoxon + BH-FDR
│       └── eval_harness.py             # load checkpoint → run 500 samples
├── scripts/
│   ├── cosmos_sanity_check.py          # tokenizer fidelity gate (Week 1)
│   ├── validate_masks.py               # mask invariants for all variants
│   ├── inspect_scene_desc.py           # visual mask inspection (SCITAS)
│   └── visualize_block_masking.py      # PNG mosaic for V1
├── tests/
│   ├── test_metrics.py
│   ├── test_scene_parser.py
│   ├── test_scene_parser_integration.py
│   ├── test_statistical_tests.py
│   ├── test_block_masking.py
│   ├── test_context_block_masking.py
│   └── test_span_masking.py
├── cfgs/nano4M/
│   ├── multiclevr_d6-6w512.yaml            (reference)
│   ├── multiclevr_d6-6w512_baseline.yaml   (V0)
│   ├── multiclevr_d6-6w512_block.yaml      (V1)
│   ├── multiclevr_d6-6w512_ctxblock.yaml   (V2)
│   └── multiclevr_d6-6w512_span.yaml       (V3)
├── run_training.py
└── run_evaluation.py                       # guarded until eval_harness is wired
```

All four variant configs send to `Com304-team11/nano4M-masking` on wandb.

---

## Status

### Week 1 — DONE (2026-04-28)

- All four masking classes shipped on `main`.
- Parser, Hungarian matcher, metrics (per-field accuracy, set_match,
  exact_sequence, AbsRel, RMSE, δ₁, angular error, SSIM, FID secondary),
  Wilcoxon + BH-FDR all in place with unit tests.
- `cd nano4M && python scripts/validate_masks.py` → 4/4 PASS
  (50/50 invariant checks per variant).
- `cd nano4M && python -m pytest tests/ -q` → 73/73 PASS.
- Cosmos-DI16x16 sanity: mean SSIM = 0.9580 on 50 CLEVR val images
  (gate ≥ 0.85 cleared).

### Week 2 — Training (next)

Queue the four runs on SCITAS as soon as the queue has slots:

```bash
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 run_training.py \
    --config cfgs/nano4M/multiclevr_d6-6w512_baseline.yaml
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 run_training.py \
    --config cfgs/nano4M/multiclevr_d6-6w512_block.yaml
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 run_training.py \
    --config cfgs/nano4M/multiclevr_d6-6w512_ctxblock.yaml
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 run_training.py \
    --config cfgs/nano4M/multiclevr_d6-6w512_span.yaml
```

While the runs are going, Ralph fills in the two stubs in
`nano4M/nanofm/evaluation/eval_harness.py` (`load()` and `run()`) so
end-to-end eval works on the first checkpoint that lands.

Monitor val loss daily on wandb; if a variant clearly underfits, extend
its budget per proposal Section VI.

### Week 3 — Ablations + evaluation

Pre-flight on a SCITAS login node:

```bash
HF_HOME=/work/com-304/hf_cache python scripts/prefetch_eval_models.py
```

Nine ablation configs to be created from the main configs by fixing a
single masking parameter:

| Variant | Ablation knob       | Singletons        |
|---------|---------------------|-------------------|
| V1      | `block_sizes`       | `[2]`, `[3]`, `[4]` |
| V2      | `context_block_sizes` | `[4]`, `[5]`, `[6]` |
| V3      | `mean_span_length`  | `2`, `3`, `5`     |

After all checkpoints finish, submit one SLURM job per (variant, seed):

```bash
for v in baseline block ctxblock span; do
  for seed in 0 1 2; do
    sbatch submit_eval_job.sh \
      outputs/nano4M/multiclevr_d6-6w512_${v}/checkpoint-final.safetensors \
      cfgs/nano4M/multiclevr_d6-6w512_${v}.yaml \
      results/${v}_seed${seed}.json \
      ${seed} \
      A,B,C,D
  done
done
```

Then aggregate + apply paired Wilcoxon + BH-FDR across the full grid:

```bash
python scripts/aggregate_eval_results.py \
    --results-dir results/ \
    --baseline-variant baseline \
    --output-dir results/
```

This emits `aggregate_table.csv`, `significance_table.csv`, and
`comparison_summary.md` (with `*` markers on q < 0.05 metrics).

### Week 4 — Report

4-page report, website, figures, presentation. No new training.

---

## Metrics

After integrating the TA's eval feedback the harness reports two families
of metrics. Each metric answers exactly one question.

### Per-modality reconstruction (proposal Section IV)

For each target modality, the encoder receives all OTHER modalities fully
visible and the model generates the target.

| Modality    | Primary                                                                                          | Secondary |
|-------------|--------------------------------------------------------------------------------------------------|-----------|
| Depth       | AbsRel                                                                                           | RMSE, δ₁ |
| Normals     | Mean angular error (degrees)                                                                     |           |
| RGB         | SSIM vs Cosmos reconstructions                                                                   | FID |
| Scene desc  | Per-field accuracy (position ±3 px, shape/color/material exact)                                  | `set_match`, `exact_sequence`, `parse_rate` |

### Cross-modal generation (TA's headline ask)

For each task, the encoder receives ONLY the source modality (other
modalities not given) and the model generates the target.

| Task                       | Primary                                              | Secondary |
|----------------------------|------------------------------------------------------|-----------|
| RGB → scene_desc           | `cross/rgb_to_text/llm_alignment` (Qwen judge)       | parser per-field accuracy, `llm_perfect_rate`, `llm_parse_error_rate`, `parser_judge_corr` |
| scene_desc → RGB           | `cross/text_to_rgb/obj_f1` (GroundingDINO)           | FID, SSIM, `obj_precision`, `obj_recall`, `obj_perfect_rate` |

Reported as mean ± std across 3 generation seeds per variant. Paired
Wilcoxon + BH-FDR across the full grid via `scripts/aggregate_eval_results.py`.

### Determinism notes

- Model generation is stochastic (τ=1.0); 3 seeds give the variation signal.
- Qwen judge is deterministic (`do_sample=False, temperature=0`).
- GroundingDINO and Cosmos decoder are deterministic forward passes.
- Val iteration is sorted-by-filename, so all variants score the same first
  N val samples.

---

## Adding a new masking class

`nanofm.data.multimodal.create_multimodal_masked_dataloader` accepts an
optional Hydra-style `masking:` dict. To add a new variant:

1. Drop the class in `nano4M/nanofm/data/multimodal/<your_module>.py`,
   inheriting from `SimpleMultimodalMasking` and overriding either
   `__call__` (V2/V3 pattern) or `perform_random_masking`
   (V1 pattern). Match the 8-key output dict.
2. Wire `masking:` into both `train_loader_config` and `eval_loader_config`
   in your variant YAML:
   ```yaml
   masking:
     _target_: nanofm.data.multimodal.<your_module>.<YourClass>
     <variant-specific kwargs>
   ```
   The shared kwargs (modalities, vocab_sizes, alphas, ranges, overlap_*)
   pass through automatically.
3. Add a registry entry in `scripts/validate_masks.py` and
   `scripts/inspect_scene_desc.py` (mirror the existing variants).
4. Add a pytest module under `tests/test_<your_variant>_masking.py`.

---

## Useful one-liners

```bash
# Full unit-test suite
cd nano4M && python -m pytest tests/ -q

# Mask invariants on all four variants
cd nano4M && python scripts/validate_masks.py

# V1 visual smoke (synthetic, runs anywhere)
cd nano4M && python scripts/visualize_block_masking.py --no-figure --num-samples 4

# What changed since this morning
git log --oneline --since="this morning"

# Pull professor's upstream into our main
git fetch upstream && git merge upstream/main && git push origin main
```

---

## Risks (from proposal Section VI)

- **V2 s=4 instability** (93.75% mask ratio): the main V2 run samples
  s ∈ {4,5,6} so the risk only applies to the s=4 ablation. Fall back to
  s=6 if it fails to converge.
- **SCITAS queue latency**: queue Week-2 jobs as soon as slots open.
- **Silent masking bugs**: `validate_masks.py` programmatically checks
  invariants; `inspect_scene_desc.py` does visual checks on real CLEVR
  samples on SCITAS.
- **Mask-ratio confound for V2**: addressed by budget-matching (regime 1
  matches V0/V1 mask count exactly at our `(1, 128)` budget).
