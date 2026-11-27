# Running the StreetCLIP fine-tuning on the IE University SLURM cluster

Runbook for model 7 of the results table (`streetclip_finetuned`). Louis prepares the cluster
(steps 1–7), Faris runs the experiments (steps 8–11). Everything is in `scripts/slurm/`.

**What we do not know yet** (nobody has told us): partition names, GPU types, accounts / QOS, the
module system, scratch paths, time limits, and whether compute nodes have internet. Step 3 finds
out; every such value then goes in ONE file, `scripts/slurm/cluster.env`. Nothing in these scripts
has been run on the real cluster yet (see "Not verified yet" at the end).

| File | What it does | Where it runs |
|---|---|---|
| `00_inspect_cluster.sh` | read-only survey: partitions, GPUs, limits, accounts, modules, disks, internet | login node |
| `cluster.env.example` | template of all site-specific values → copy to `cluster.env` | – |
| `setup_env.sh` | creates the conda env (or a venv) and installs the package + `peft` | login node |
| `fetch_assets.sh` | downloads the Kaggle dataset and StreetCLIP, checks them | login node (needs internet) |
| `submit.sh` | builds and runs the `sbatch` command from `cluster.env` (`DRY_RUN=1` = only print) | login node |
| `smoke.sbatch` | ≤ 15 min test: GPU, data, offline model, checkpoints, resume | GPU node |
| `finetune.sbatch` | one training run, always with `--resume` | GPU node |
| `export.sbatch` | exports logits from `best.pt` (val; test only with `FINAL_TEST=1`) | GPU node |
| `common.sh` | shared helpers (read `cluster.env`, load modules, activate env) | sourced |

## Who does what

| Step | Louis (setup, checkpoints, comparisons) | Faris (trainings, experiments) |
|---|---|---|
| 1–4 | ssh, clone, inspect the cluster, fill `cluster.env` | review `cluster.env` with Louis |
| 5–6 | `setup_env.sh`, `fetch_assets.sh` | – |
| 7 | smoke test, write the measured time / memory in this file | read the smoke log |
| 8–9 | check checkpoints and disk use | submit the real runs, monitor them |
| 10 | resume runs cut by the time limit | decide which runs to continue |
| 11 | copy results back, comparison notebook (`07`) | experiments notebook (`06`), choose the final run on **val** |
| end | – | final export with `FINAL_TEST=1` (once) |

## 1. Connect

```bash
ssh <your-ie-user>@<cluster-login-host>     # host name: from the IT / course instructions
```

## 2. Clone the repo (on the cluster)

```bash
git clone <repo-url> geosense
cd geosense
git checkout louis-faris/streetclip-finetune
```

All commands below are run **from the repo root**.

## 3. Inspect the cluster (read-only)

```bash
bash scripts/slurm/00_inspect_cluster.sh 2>&1 | tee inspect.txt
```

It changes nothing. Send `inspect.txt` to the team (do not commit it). The sections are named
`[sinfo]`, `[sacctmgr assoc]`, `[module avail ...]`, `[disk ...]`, `[internet from the LOGIN node]`, …

## 4. Fill `cluster.env`

```bash
cp scripts/slurm/cluster.env.example scripts/slurm/cluster.env
nano scripts/slurm/cluster.env
```

Each value has a comment saying which section of `inspect.txt` gives it. The important ones:
`GS_PARTITION` (a partition with `gpu` in its GRES), `GS_ACCOUNT` / `GS_QOS` (leave empty to use your
defaults: empty values are not passed to `sbatch`), `GS_GRES`, `GS_TIME` (≤ the partition limit),
`GS_MODULES` (a conda module if conda is not already on the PATH), and the paths `GEOSENSE_DATA`,
`GEOSENSE_CKPT`, `HF_HOME` (on scratch / a big disk, not in the repo). Never put a password or a
Kaggle key in this file. Check what will be submitted, without submitting:

```bash
DRY_RUN=1 scripts/slurm/submit.sh --smoke
```

## 5. Create the Python environment (login node, once)

```bash
bash scripts/slurm/setup_env.sh
```

Creates the conda env `GS_CONDA_ENV` (or a venv in `GS_VENV_DIR` if there is no conda), installs
the package with `pip install -e ".[dev]"`, plus `peft` and `kagglehub`. Safe to re-run. PyTorch
comes from pip: the Linux wheel contains its own CUDA, so it works on the GPU nodes even though
`cuda available: False` is printed on the login node.

## 6. Download the data and the model (login node, once)

```bash
bash scripts/slurm/fetch_assets.sh
```

- **Dataset** (6.7 GB) with `kagglehub` into the parent of `GEOSENSE_DATA`. Kaggle may ask for your
  API token: create it on kaggle.com (Settings → API → Create New Token), then either copy
  `kaggle.json` to `~/.kaggle/kaggle.json` yourself (`chmod 600 ~/.kaggle/kaggle.json`), or type it
  into the current shell only:
  ```bash
  read -rp "Kaggle user: " KAGGLE_USERNAME; read -rsp "Kaggle key: " KAGGLE_KEY; export KAGGLE_USERNAME KAGGLE_KEY
  ```
  **Alternative without Kaggle:** copy the folder from a laptop that already has it, then only check:
  ```bash
  # on the laptop (trailing slashes: copy the CONTENT of compressed_dataset into GEOSENSE_DATA)
  rsync -avP ~/Downloads/geosense_dataset/compressed_dataset/ <user>@<login-host>:<GEOSENSE_DATA>/
  # on the cluster
  bash scripts/slurm/fetch_assets.sh --check
  ```
- **StreetCLIP** (1.7 GB) into `HF_HOME`, so the GPU jobs run with `HF_HUB_OFFLINE=1` (compute nodes
  often have no internet).
- **Checks**: 49,997 JPEGs; every `path` of `data/splits.csv` exists; StreetCLIP files load offline.

## 7. Smoke test (the FIRST job, ≤ 15 min)

```bash
scripts/slurm/submit.sh --smoke
squeue -u $USER
tail -f logs/geosense-smoke-<jobid>.out
```

It runs `--subset 500 --epochs 1 --method last_blocks --unfreeze-blocks 1` (500 train images,
200 val images), then the same command with `--epochs 2 --resume`, and checks that the history has
epochs `0,1,2` (0 = zero-shot reference before training). The log ends with `SMOKE TEST PASSED` or
`FAILED`, the time of each pass, the peak GPU memory (from `logs/<job>-gpu.csv`) and whether the
compute node has internet. **Write those numbers in the "Estimates" section below.** If it is
killed by the time limit on a slow GPU, set `GS_SMOKE_TIME="00:30:00"`.

## 8. Real runs (Faris)

```bash
scripts/slurm/submit.sh <run_name> [geosense.finetune options]

scripts/slurm/submit.sh ft_head   --method head --epochs 3                    # sanity: ≈ linear probe
scripts/slurm/submit.sh ft_lb2    --method last_blocks --unfreeze-blocks 2    # default method
scripts/slurm/submit.sh ft_lora8  --method lora --lr 1e-4 --lora-r 8
```

The job always adds `--ckpt-dir $GEOSENSE_CKPT --resume` and uses `cores - 1` DataLoader workers.
Each run writes `results/<run>_history.csv`, `results/<run>_config.json`,
`$GEOSENSE_CKPT/<run>/last.pt` + `best.pt`, and exports the validation logits from `best.pt` to
`predictions/runs/<run>_val_logits.npy` (never the team file). One run name = one experiment: use a
new name when you change the options.

## 9. Monitor

```bash
squeue -u $USER                                   # PD = waiting, R = running
tail -f logs/<run_name>-<jobid>.out               # one line per epoch: val_acc, val_macro_f1, seconds
cat results/<run_name>_history.csv
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode   # after the end
scancel <jobid>                                   # stop a job (checkpoints of finished epochs stay)
```

## 10. Resume after a time limit / crash

A checkpoint is written after every epoch. If the log ends with `DUE TO TIME LIMIT` (or the node
crashed), submit **exactly the same command again**:

```bash
scripts/slurm/submit.sh ft_lb2 --method last_blocks --unfreeze-blocks 2
```

The log then says `resumed after epoch N`; at most one epoch is lost. Optional: `GS_AUTO_REQUEUE="1"`
in `cluster.env` makes SLURM warn the job 2 min before the limit (`--signal=B:USR1@120`) and the
job requeues itself (`--requeue`). Only use it if one epoch is clearly shorter than `GS_TIME`.

## 11. Bring the results back

Copy to a laptop (or commit from the cluster) **only these files**:

| Copy / commit | Never |
|---|---|
| `results/<run>_history.csv`, `results/<run>_config.json` | `*.pt` / anything in `GEOSENSE_CKPT` (checkpoints) |
| `predictions/runs/<run>_val_logits.npy` (experiments; git-ignored, share them or copy for notebook 07) | images, the dataset, `HF_HOME` |
| at the very end: `predictions/streetclip_finetuned_val_logits.npy` and `_test_logits.npy` | `logs/`, `inspect.txt`, `cluster.env` |

```bash
# on the laptop, from the repo root
rsync -avP <user>@<login-host>:<cluster-repo>/results/ results/ --include='*_history.csv' --include='*_config.json' --exclude='*'
rsync -avP <user>@<login-host>:<cluster-repo>/predictions/runs/ predictions/runs/
```

**Final export (once, at the end, for the ONE run chosen on validation):**

```bash
scripts/slurm/submit.sh --export ft_lb2                     # validation only -> predictions/runs/
scripts/slurm/submit.sh --export ft_lb2 --final             # validation only -> the team contract file (val)
FINAL_TEST=1 scripts/slurm/submit.sh --export ft_lb2        # val + TEST -> the team contract files
```

`FINAL_TEST=1` prints a warning and asks you to type the run name. Team rule: the test set is used
once, at the end, never to choose between runs. Then commit the two
`predictions/streetclip_finetuned_*_logits.npy` files.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | lower `--batch-size` and raise `--accum` by the same factor (same effective batch): `--batch-size 16 --accum 2`, then `8 / 4`. LoRA needs much more memory than `last_blocks` (see below) |
| `ERROR: .../cluster.env not found` | step 4 |
| Job stays `PD` for hours | `squeue -u $USER -l` shows the reason (`Resources`, `Priority`, `QOSMaxWallDurationPerJobLimit`…): ask for less time (`GS_TIME`), another partition, or a smaller `GS_MEM` |
| `Invalid account or account/partition combination` | `GS_ACCOUNT` / `GS_PARTITION` from `[sacctmgr assoc]`; try leaving them empty |
| `Invalid generic resource (gres) specification` | `GS_GRES` must match `[sinfo GRES per partition]` (`gpu:1` or `gpu:<type>:1`) |
| No log file appears | the `logs/` folder must exist (`submit.sh` creates it; with plain `sbatch` run `mkdir -p logs`) |
| `OSError: ... couldn't connect to huggingface.co` / `LocalEntryNotFoundError` | compute nodes have no internet: run `fetch_assets.sh` on the login node with the SAME `HF_HOME` |
| `Disk quota exceeded` | `GEOSENSE_DATA`, `GEOSENSE_CKPT`, `HF_HOME` and the conda env (`GS_CONDA_ENV` can be a path) on scratch / a project disk, not `$HOME`; delete `smoke_*` checkpoints; `pip cache purge` |
| Scratch files disappeared | many clusters purge scratch after N days: keep only checkpoints there, and copy `results/` + `predictions/runs/` back after each run |
| `torch ... cuda available: False` in a GPU job | the job did not get a GPU (`GS_GRES`), or the driver is too old for the wheel: check `nvidia-smi` "CUDA Version" and set `GS_TORCH_INDEX_URL` (e.g. `.../whl/cu118`), re-run `setup_env.sh` |
| Very slow epochs, GPU utilisation low in `logs/<job>-gpu.csv` | the data loading is the bottleneck: more `GS_CPUS`, or copy the dataset to fast local disk |

## Estimates — to be REPLACED by the smoke-run measurement

These are back-of-the-envelope numbers, not measurements. The smoke log gives the real ones.

- Model: CLIP ViT-L/14 at 336 px, ~304M parameters in the vision tower (+ projection), 577 tokens per
  image. ~1.2 GB of fp32 weights on the GPU (the text tower is dropped after building the head).
- Data per epoch: 33,464 training images + 7,171 validation images (evaluation every epoch,
  plus epoch 0 before training).
- Compute: ~190 GFLOPs per image forward. `last_blocks`: the frozen blocks run forward only, so one
  epoch ≈ 41k forward passes ≈ 8 PFLOP. `lora`: back-propagation through all 24 blocks, ≈ 3× more.

| | `head` / `last_blocks` (N = 2) | `lora` (r = 8) |
|---|---|---|
| Trainable parameters | head 43k / ~25M | ~0.8M |
| GPU memory, batch 32, AMP (rough) | ~6–10 GB | ~40–60 GB → use `--batch-size 8 --accum 4` on a 24 GB GPU |
| Time per epoch, A100 (rough) | ~5–10 min (often limited by JPEG decoding: give ≥ 8 cores) | ~15–25 min |
| Time per epoch, V100 / T4 (rough) | ~15–40 min | ~1–2 h |
| Full run (≤ 8 epochs, early stopping) | ~1–4 h | ~3–12 h |

Measured on the cluster (fill in after step 7):

| GPU | pass 1 time (s) | peak GPU memory (MiB) | epoch time of a real run | date |
|---|---|---|---|---|
| … | … | … | … | … |

## Not verified yet

Written and tested on a laptop only (no SLURM there): `bash -n` on every script, `submit.sh` with
`DRY_RUN=1`, and `smoke.sbatch` executed as a plain bash script with a tiny random model on CPU.
Still unverified until someone runs them on the cluster: the real `sbatch` submission and the
partition / account / gres values, the module and conda activation on the nodes, the GPU, the
offline StreetCLIP loading, the Kaggle download, the SIGUSR1 requeue, and all the time / memory
numbers above.
