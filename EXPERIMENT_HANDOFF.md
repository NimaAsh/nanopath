# Nanopath Experiment Handoff — 2026-06-05 (post-merge, Compute Canada)

Continuation of the Alliance loop to raise `mean_probe_score`, after `MedARC-AI:main` merged into
`compute-canada`. Covers four experiment waves and the cluster/tooling changes. All numbers are on the
**new (leopard) probe suite** — the merge swapped the survival probe `boehmk_pfs → leopard_bcr`, which runs
~+0.007 higher than the old handoff's numbers, so **do not compare across the merge**.

## TL;DR

- **NEW BEST: `cc-main-jepa` = 0.6481** — DINO + an **I-JEPA predictor** replacing iBOT. **+0.0105** over the
  prior iBOT champion (0.6376), broad gains with classification held. **First thing to break the ~0.637
  ceiling** after a long null streak. Single seed → **replicating** (seeds 2026/4099) before submitting.
- If it holds it **tops the leaderboard** (the prior champion was ~tied with the public leader ~0.634, so
  JEPA is ~+0.014 over the competitor — well past the +0.006 bar).
- Everything else this session was null/negative: schedule re-key, seg-objective knobs, tissue curation,
  ViT-B capacity, stain normalization.

## Ground rules

- Improvement counts only at **+0.006** `mean_probe_score`. Seed SD on the new suite ≈ 0.0018.
- To **top** the board: beat the highest validated run by +0.006; the maintainer **reruns with a fresh seed**,
  so the gain must hold in expectation (replicate; don't cherry-pick a lucky seed).
- Caps: `max_train_samples == 1e6`, `tile_presentations <= 1e6`, `max_train_flops == 1e18`.
  Locked paths: `probe.py`, `benchmarking/` (and `probe.*` config vars except `dataset_roots`).

## Scoring leverage map

`mean_probe_score` = unweighted mean of **8 task-type slots**: `linear_f1, knn_f1, fewshot_f1, seg_jaccard,
slide_auc(progression), auc(mutation), survival_cindex, robustness`. The 4 classification datasets feed 3 of
8 slots → classification ≈ 37% of the score. `seg_jaccard ≈ 0.31` is the floor.

## Champion progression

| recipe | score | note |
|---|---:|---|
| stock cc-main (ViT-S) | ~0.617 | baseline |
| gentleaug-local112 (iBOT) | 0.6376 | gentler aug + 112px locals; prior champion (3-seed ~0.636) |
| **jepa (DINO + I-JEPA)** | **0.6481** | **+0.0105; new best, replicating** |

JEPA per-slot Δ vs iBOT champion (broad, classification held):
`lin +0.001 · knn +0.003 · few +0.024 · seg −0.001 · slide +0.033 · auc −0.012 · surv +0.029 · rob +0.008`.
The win is on **global/holistic slots (slide, fewshot, survival)**, NOT segmentation. Robust to the one noisy
slot: even if slide (single ds, σ~0.07) fully reverted, JEPA ≈ 0.644 = still +0.006. Also **faster** (37 vs
47 min — the 4-block predictor is cheaper than the 131072-prototype iBOT head + Sinkhorn).

---

## Wave 1 — Schedule re-key + first sweep (9 runs) → NULL, reverted

Re-keyed the (flop-bound) schedules to the sample cap so LR anneals/KDE ramps complete. Net **−0.0028**
(traded linear/robustness for fewshot/seg); reverted. Seg-objective knobs (iBOT weight 1.5, mask_prob 0.7,
blur 0.1) all **lowered** seg. local126/views10/warmup03-05 all null. Nothing cleared +0.006.

## Wave 2 — Tissue curation port (4 runs) → NULL

Ported a competitor's recipe (tissue rejection 0.5 + samples-keyed warmup + adam_beta2 0.99). On gentleaug it
**hurt −0.0026**; the faithful stock-aug repro = 0.6343 (below our champion). **Gentleaug and tissue-curation
are substitutes, not complements** — same ~0.637 ceiling, don't stack. Kept knobs at neutral defaults.

## Wave 3 — ViT-B capacity (1 base + 4-config recovery sweep) → DEAD END

ViT-B (86.6M) = 0.6310. A profile shift: wins fewshot/seg/robustness, collapses linear/slide (underfit at
1M tiles). Recovery sweep — lwd {0.5, 0.85}, lr 5e-5, views 8 — all 0.628–0.631, none beat the original ViT-B
or the ViT-S champion. **Linear/slide are anti-correlated under these knobs; no sweet spot.** ViT-S is the
1M-tile sweet spot; capacity is exhausted. `layerwise_decay` direction (for the record): LOWER decay freezes
early layers harder (preserve pretrained), HIGHER adapts more.

## Wave 4 — Masked-latent objectives → JEPA WINS

Inspired by GenBio-PathFM's JEDI (JEPA+DINO). Implemented a config-gated patch objective: `dino.jepa_weight>0`
swaps iBOT for an I-JEPA predictor (`model.py JEPAPredictor`, shallow transformer) that predicts the EMA
teacher's masked-patch reps; `dino.jepa_cluster` chooses the target.

| run | score | vs champ | note |
|---|---:|---:|---|
| **jepa** (predictor + smooth-L1 regression, block masks) | **0.6481** | **+0.0105** | new best |
| capi (predictor + iBOT Sinkhorn clusters, random masks) | pending | — | OOM bug fixed (was block-masking the 131072-proto head); re-running |
| stainnorm (Macenko stain norm, iBOT) | 0.6323 | −0.005 | another nuisance-suppression substitute — null |

---

## In flight / queued

- **JEPA seed replicates** — `cc-main-jepa-seed2026`, `cc-main-jepa-seed4099` (priority #1; confirm +0.006 holds).
- **CAPI** — `cc-main-capi` (predictor + clustered targets; may beat plain JEPA).
- **JEPA variant sweep** (push past 0.6481): `cc-main-jepa-w2` (weight 2.0), `-whalf` (0.5),
  `-deep` (predictor depth 6, via `dino.jepa_pred_depth`), `-curate` (tissue 0.5 re-tested on JEPA).

All write to `/scratch/$USER/nanopath/main-leopard/`; launch with `./submit/cc_train_1gpu.sbatch configs/<cfg>.yaml`
(any Alliance cluster, self-submits). Smoke check: first ~2 min, the `ibot:` log field is the JEPA/CAPI patch
loss — finite + steps advancing = wired right. Pack with `nppackruns N`, read slots with `npscores`.

## Patch-objective modes (one flag pair)

| mode | jepa_weight | jepa_cluster | student path | target | masking |
|---|---|---|---|---|---|
| iBOT (champion) | 0 | — | head | Sinkhorn clusters, CE | random |
| **JEPA (best)** | >0 | false | predictor | raw features, smooth-L1 | block |
| CAPI | >0 | true | predictor + head | Sinkhorn clusters, CE | random |

## Code / infra this session

- **JEPA/CAPI** in `model.py` (`JEPAPredictor`) + `train.py` (gated patch objective, `make_block_mask`,
  `patch_modules` handling). Config knobs `dino.jepa_weight / jepa_cluster / jepa_pred_depth` (lazily read —
  iBOT configs don't need them). JEPA/CAPI require `save_every: null` (no iBOT heads to checkpoint under JEPA).
- Earlier knobs (all neutral defaults, champion unchanged): `data.blur_prob`, `data.tissue_thresh`,
  `data.stain_norm` (Macenko `StainNorm`), `dino.ibot_loss_weight`, `dino.mask_prob`, `dino.adam_beta2`.
  Warmup is sample-keyed (frac stays flop-keyed).
- `submit/cc_train_1gpu.sbatch` — unified self-submitting launcher (`$CC_CLUSTER`); job name `nanopath`.
- `prepare.py` localize fix (writable-ancestor test, so it no longer rewrites `/scratch` output_dirs).
- `submit/cc_helpers.sh` — version-controlled cluster shell helpers (source from each `.bashrc`).
- Per-cluster setup (Alliance `/scratch` is per-cluster): on each login node, `uv sync` then
  `export TORCH_HOME/HF_HOME=/scratch/...` then `python prepare.py configs/cc-main-vitb.yaml download=True`.

## Submit the winner (after replication)

```bash
git status --short probe.py benchmarking      # must be empty
./labless/submit_to_labless.py output_dir=/scratch/$USER/nanopath/main-leopard/jepa \
  run_name=jepa-dino notes="DINO + I-JEPA predictor (smooth-L1 latent regression) replacing iBOT; ViT-S, gentleaug+local112."
```
The score is read from `summary.json` (don't put it in the notes); notes describe the change.

## Suggested next steps

1. Confirm JEPA replicates (seeds 2026/4099) → **submit it** (tops the board by a wide margin).
2. See if CAPI beats plain JEPA (clustered vs regression targets).
3. Push past 0.6481 with the variant sweep; if a direction helps, combine (e.g. best weight + depth), then
   consider dual-stage JEDI (JEPA→DINO) and re-testing aug/curation on the JEPA base.

## Falsified — do not re-litigate (on the new suite, ViT-S/1M)

Full schedule re-key (frac→samples); iBOT-weight/heavier-masking/softer-blur for segmentation; tissue
curation stacked on gentleaug; adam_beta2 0.99; bare warmup-fraction sweeps; ViT-B capacity (base + recovery
sweep); stain normalization. All null-to-negative.
