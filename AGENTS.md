Project goals:
- Nanopath should be easy to share with collaborators so they can quickly try new training objectives, preprocessing choices, data curation ideas, and hyperparameters on a small model.
- The normal loop is: iterate fast on a single H100, validate promising changes with downstream probes, then reserve larger/full-node training for the best candidates.
- Keep the codebase hackable and nanochat-like: flat organization, few files, few lines, and minimal abstractions.

Before changing code:
- If not already activated, source nanopath's .venv.
- For broad or ambiguous tasks, read deeply enough into the current repo to understand the training/probing/data path before recommending changes. Look at every relevant source, config, script, and doc file rather than optimizing one file in isolation.
- Make a concrete multi-step plan for nontrivial work, then keep going through implementation, validation, and any needed doc/comment updates.
- Default to immediately implementing sensible recommendations and validating them rather than simply suggesting recommendations.

Coding guidelines:
- Use flat organization: as few folders, files, and lines of code as possible; functional elegance is the goal. If you make revisions where there are over a dozen new lines of code I am going to be highly skeptical you really tried your best to adhere to this guideline. A great revision should LOWER the total lines of code, not increase it. Don't play smart by opening subprocesses or other hacks to get around this limitation.
- Commenting is the exception to the line-count preference: add concise comments explaining "how and why" for functions and important (i.e., not plotting/logging) code blocks.
- Do not add defensive `try`/`except` blocks or fallbacks. If something is wrong, it should fail loudly. Don't bother with ValueError raises or other code guards.
- Prefer hard-coded constants over extra environment variables, modular options, or fallback paths, unless the value is meant to be frequently tuned.
- Prefer native PyTorch over Accelerate, Lightning, etc. nanopath is single-GPU only — do not introduce DDP, FSDP, or any multi-GPU code path.
- Do not use `argparse`. Meaningful tunables should live in YAML config files, e.g. `cfg.train.lr`; if YAML does not define a variable used by a training script, it is fine for that to error. Only put variables in YAML when they are actually meant to be tuned often; otherwise hard-code them.
- Avoid tiny helper functions/classes that are only a handful of lines. Put the code directly where it is used.
- Follow [nanochat](https://github.com/karpathy/nanochat) as the model for a clean minimalist codebase, especially `train.py` and `model.py`.
- Do not create new files unless explicitly asked or truly necessary; prefer improving existing files. If you do create a new file, add a few commented out lines of code to the top of it to explain its purpose.
- If code changes make comments, docs, configs, or scripts inaccurate, update those too.

Experiment and benchmark discipline:
- Validate opinions experimentally whenever feasible. Run code, tests, probes, or short jobs that directly support the conclusion.
- Use downstream probing as the main comparison signal because objectives like JEPA, MAE, DINO, and iBOT may not have comparable validation losses.
- An improvement should only actually be considered an improvement when mean_probe_score improves by at least .006; anything less is within random variance.
- Use wandb for logging, plotting, and utilization monitoring throughout pretraining. Log all metrics needed to validate training behavior (i.e., gradient norm).
- Aim for >80% GPU utilization during GPU runs; investigate and remedy code when utilization is poor.
- Full runs launched with `./submit/train_1gpu.sbatch ...` prompt for Labless run name, notes, and GitHub no-scope device login before scheduling, then auto-submit after a successful eligible run. For direct `python train.py` runs or frozen baseline evaluations worth sharing, run `./labless/submit_to_labless.py output_dir=... run_name=... notes=...`; labless records the verified GitHub login, and each login can submit at most 20 runs per 24 hours. Full submissions require `summary.json`, `metrics.jsonl`, `summary.max_train_samples == 1000000`, `summary.tile_presentations <= 1000000`, and `summary.max_train_flops == 1e18`. Keep smoke checks and failed runs local.
- Do not submit runs whose saved `labless_source` snapshot changes `probe.py` or `benchmarking/`; labless marks locked-path changes invalid.

Cluster and storage:
- Current branch target is Compute Canada/Alliance Rorqual and Trillium, not Nebius or the Sophont/MedARC cluster. Do not use Sophont-specific `#SBATCH --partition=n`, `/data`, `/block`, or direct `ssh n-#` workflows.
- Alliance login nodes have internet but no GPU access; compute nodes have GPU access but no internet. Run `prepare.py ... download=True` only on the login node, then verify with `download=False` before submitting GPU jobs.
- Store large files, checkpoints, embeddings, HF/Torch/W&B caches, probe data, and pretrained models under `/scratch/$USER/nanopath` (or a project allocation if explicitly chosen), not the repo or `$HOME`. Configs should use literal `$USER`, expanded by `train.py` at load time.
- Use `configs/cc-smoke.yaml`, `configs/cc-main.yaml`, and `configs/cc-main-fastwarmup.yaml` for Alliance runs. Launch Rorqual with `submit/cc_train_1gpu.sbatch` and Trillium with `submit/trillium_train_1gpu.sbatch`; jobs run W&B offline on compute nodes.
- Request enough CPUs for DataLoader/probe workers; Alliance defaults can be too low for configs with `train.num_workers: 16`. Do not add unnecessary lines like `OMP_NUM_THREADS` or `MKL_NUM_THREADS` exports.
- Fresh launches should overwrite any existing `project.output_dir` unless `train.resume` is set.

Workflow:
- Do not stop after the first small fix on a difficult ask. Continue through the adjacent tasks needed to make the change credible, such as config updates, probes, throughput checks, README notes, or cleanup.
- Use parallel agents, git worktrees, or independent jobs when they materially speed up exploration or experiments, but keep changes easy to review. Make sure to kill hanging or no longer useful sub-agents.
