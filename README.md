# PI-Mamba: Physics-Informed State Space Models for Protein Design

This repository contains the implementation of **PI-Mamba**, corresponding to the paper "PI-Mamba: Scalable Protein Backbone Generation via Physics-Informed Spectral Initialization".

+ We are aware there are some scripts not the latest version, which will be released soon!

PI-Mamba generates protein CA traces with flow matching. Its sequence encoder is a stack of state-space layers whose state transition is fixed by the Rouse spectrum of a polymer chain, `A_n = exp(-lambda_n * dt / tau)`, with input-dependent `tau` and `dt`.

## Repository Structure

- `src/`: Model, data and evaluation code (a Python package, imported as `src`)
  - `backbone_pi_mamba.py`: PI-Mamba backbone, shared flow-matching base class and sampler
  - `pi_mamba_layer.py`: PI-Mamba layer (Rouse-derived state transition, vectorised scan)
  - `rouse_physics.py`: Rouse eigenvalues, mode transform, spectrum variants
  - `backbone_performer.py`, `backbone_transformer.py`: baseline encoders with the same heads and sampler
  - `geometry.py`: NeRF, torsions, Kabsch superposition, CA-to-backbone placement
  - `data.py`: PDB reading and writing, dataset, length-bucketed batching
  - `losses/`: auxiliary structure losses on CA coordinates, including
    `register_loss.py` (beta-sheet register; off by default, and its docstring
    records what it did and did not do in training)
  - `metrics.py`: CA-only structure metrics and TM-score
  - `designability.py`: scTM pipeline (ProteinMPNN + ESMFold)
  - `inpainting.py`, `benchmark_scaffold.py`: motif scaffolding
  - `measure_mode_concentration.py`, `validation_callback.py`: Rouse mode concentration C_k
  - `model_io.py`: model construction and checkpoints
- `scripts/`: Training, evaluation and experiment scripts
  - `train.py`: Main training script
  - `evaluate.py`: Sample generation, structure metrics, scTM evaluation
  - `validate_physics.py`: Physics validation (Figure 4)
  - `diagnose_sctm.py`: per-sample structural features and their correlation with scTM
  - `sheet_quality.py`: beta-strand pairing and register of a set of backbones
  - `novelty_check.py`: nearest-neighbour TM-score of samples against the training set
  - `sweep_sampling.sh`, `collect_sweep.py`: sampler settings against scTM
  - `sheet_scaling.sh`: sheet formation across checkpoints
  - `sheet_recovery.py`: whether the model keeps a sheet it is handed part way along the flow path
  - `compare_recovery.py`, `compare_arms.sh`: paired comparison of two checkpoints' register retention
  - `loss_vs_register.py`: per-structure flow-matching error against register retention
  - `check_pair_dilution.py`: how much of the distance map carries a sheet, and the model's error on it
  - `check_relax_effect.py`: what the end-of-sampling relaxation does to beta ladders
  - `run_counterfactual_experiments.py`: Counterfactual experiments (Appendix S16)
  - `run_ck_sweep.py`, `run_architecture_sweep.sh`: architecture comparison
  - `run_length_sweep.py`: Rouse vs Learned-A as a function of chain length
  - `loss_weight_sweep.py`, `run_ablation_sweep.sh`: hyper-parameter sweeps
  - `run_performer_baseline.py`: runtime and memory benchmark
- `configs/`: YAML configuration files (defaults of `train.py`)
- `tests/`: Unit tests
- `PImamba_gallery/`: Six PI-mamba proteins of length 100 as full-backbone PDB files with a rendered image (see its `readme.md`).
- `PImamba_L100/`: PI-mamba 100 proteins with Length=100 and scTM~ 0.91.
- `Distilled_trainset/`: Distilled training set with scTM >= 0.95 from Proteina, 6816 CA-only structures of length 50 to 300 (70% of the full training set).

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**

- PyTorch 2.1+
- Einops, PyYAML, NumPy, SciPy, pandas, matplotlib
- Optional, for scTM evaluation: `transformers` (ESMFold) and a checkout of [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)

Run the scripts from the repository root. Check the installation with:

```bash
python tests/test_core.py
```

## Dataset

Training reads single-chain PDB files and uses the CA atoms. `Distilled_trainset/` is used by default; point `--data_dir` to another folder of `.pdb` files (e.g. CATH 4.2) to train on different data.

All chains in a batch have the same length, because the Rouse mode basis depends on the chain length. Chains are grouped by length automatically. A data set with many distinct lengths gives small batches for rare lengths.

## Training

To train the full PI-Mamba model (Paper Config: L=16, D=512):

```bash
python scripts/train.py --use_physics True --data_dir /path/to/pdb/files
```

To train the ablation baseline (Learned-A):

```bash
python scripts/train.py --use_physics False
```

A smaller model for a quick run (about 13M parameters, 2 to 3 GB of GPU memory):

```bash
python scripts/train.py --d_model 256 --n_layers 8 --max_steps 30000 --checkpoint_dir checkpoints/small
```

Defaults come from `configs/model/pi_mamba_v12.yaml` and `configs/train/single_gpu.yaml`. Any key can be overridden with a command-line flag or with an extra YAML file (`--config my.yaml`). `python scripts/train.py --help` lists all flags.

Training details:

- Flow matching on CA coordinates: `x_t = (1 - t) x_0 + t x_1` with Gaussian `x_0`, target velocity `x_1 - x_0`. Coordinates are divided by `coord_scale` (10 Å) inside the network.
- Auxiliary losses on the predicted clean structure (local distances, bond lengths, long-range distances) for `t >= 0.5`. Weights are the `lambda_*` keys.
- Random rotation augmentation, AdamW, linear warm-up and cosine decay, EMA weights.
- `checkpoint_dir` receives `best_model.pt` (lowest validation loss), `last.pt` and `train_log.jsonl`. Resume with `--resume <checkpoint_dir>/last.pt`.
- Other encoders: `--model transformer` or `--model performer`.
- Spectrum controls: `--spectrum scrambled` gives the state channels modes drawn at random from the whole Rouse spectrum instead of its slow end. `--spectrum interpolated --spectrum_alpha 0.5` mixes the Rouse rates with random rates of the same range. Reordering the same rates among state channels would not change the model, because the state projections are learned.
- `--use_physics False` makes the rates free parameters, initialised uniformly over the range of the Rouse rates of a 128-residue chain.

The torsion head (phi, psi, omega, used for NeRF full-backbone reconstruction) is trained only with `--lambda_torsion > 0`, which needs N and C atoms in the PDB files. The distilled training set is CA-only, so with it the model is a CA-trace generator and the N, C and O atoms of full-backbone PDB files are placed around the CA trace geometrically.

## Sampling and Evaluation

```bash
# 1. Generate 100 backbones of length 100
python scripts/evaluate.py --mode generate --checkpoint checkpoints/best_model.pt \
    --n_samples 100 --length 100 --eval_dir eval_output

# 2. CA-only structure metrics, compared with the training set
python scripts/evaluate.py --mode metrics --eval_dir eval_output --reference_dir Distilled_trainset

# 3. Self-consistency: ProteinMPNN (CA-only) -> ESMFold -> TM-score
python scripts/evaluate.py --mode eval --eval_dir eval_output --proteinmpnn_dir /path/to/ProteinMPNN
```

Step 2 needs no external tools. Step 3 designs 8 sequences per backbone at temperature 0.1, folds them with ESMFold, and reports scTM (best TM-score over the 8 sequences), scRMSD and pLDDT in `info.csv` and `summary.json`. Steps 2 and 3 work on any directory of PDB files placed in `<eval_dir>/pdbs`.

### Diagnostics

A mean scTM says how designable a set of backbones is, not which part of it is
weak. Three scripts break the number down, all from CA coordinates on CPU:

```bash
# Which features separate a high-scTM sample from a low-scTM one, and how the
# feature distributions compare with another set
python scripts/diagnose_sctm.py --eval_dir eval_output \
    --compare PImamba_L100 --compare Distilled_trainset

# Are the strand-labelled residues paired and in register, or extended coil?
python scripts/sheet_quality.py --samples eval_output/pdbs \
    --samples Distilled_trainset --length 100 \
    --sctm_from eval_output/info.csv

# Highest TM-score of every sample against the training structures of the same
# length, to tell designability from proximity to the training data
python scripts/novelty_check.py --samples eval_output/pdbs \
    --reference Distilled_trainset
```

`scripts/sweep_sampling.sh` runs a list of sampler settings through generation,
CA metrics and scTM and collects them into one table with
`scripts/collect_sweep.py`. It reports secondary-structure content and sample
diversity next to scTM, because scTM on its own rises when the output collapses
onto helical bundles.

Two more scripts work on the velocity field rather than on finished samples,
which is a far more sensitive test of beta-sheet quality and needs neither
ProteinMPNN nor ESMFold:

```bash
# Noise a training structure part of the way back along the flow-matching path
# and integrate to t=1: does the model keep a sheet it is handed?
python scripts/sheet_recovery.py --checkpoint checkpoints/run/last.pt \
    --reference Distilled_trainset --n 200 --t0 0.9

# The same measurement on two checkpoints is paired structure by structure,
# so differences smaller than the spread between structures are visible
python scripts/compare_recovery.py a.json b.json --key t0=0.9

# Both at once for two training arms
scripts/compare_arms.sh baseline.pt candidate.pt tag 200
```

Two scripts ask whether the loss sees the sheets at all:

```bash
# Per-structure flow-matching error against register retention
python scripts/loss_vs_register.py --checkpoint checkpoints/run/last.pt --n 200

# What fraction of the distance map carries a sheet, and whether the model is
# worse on those pairs than on the map as a whole
python scripts/check_pair_dilution.py --checkpoint checkpoints/run/last.pt --n 200
```

A note on reading scTM. At 100 samples the spread between seeds is larger than
most effects worth measuring: one checkpoint gave 0.718 and 0.785 under
identical settings. scTM also does not track sheet register, so a run with
2.7 times the beta ladders can score lower. Quote it with its seed spread and
read `registered_frac` beside it.

## Motif Scaffolding

Generate chains that contain a motif cut out of an existing structure:

```bash
python -m src.benchmark_scaffold --ckpt checkpoints/best_model.pt \
    --motif triad:Distilled_trainset/sample_000000_L100_scTM0.965.pdb:20-28 \
    --n_samples 8 --out benchmark_results.csv
```

A motif is written `NAME:PDB:RANGES[:LENGTH]` with 1-based inclusive residue ranges. During sampling the motif positions are held at the value the flow expects at the current time and only the scaffold is integrated. The report gives the motif RMSD, the clash fraction and, with `--proteinmpnn_dir`, scTM. Scaffolds produced by other methods can be scored with the same metrics through `--baseline NAME=DIR`.

The final bond projection makes all CA-CA distances exactly 3.8 Å but moves the motif slightly; `--no_final_projection` keeps the motif exact instead.

## Reproducing Results

1. **Physics Validation** (Figure 4):

   ```bash
   python scripts/validate_physics.py --checkpoint checkpoints/best_model.pt
   ```

2. **Counterfactual Experiments** (Appendix S16):

   ```bash
   python scripts/run_counterfactual_experiments.py --experiment all --seeds 0,1,2
   ```

3. **Architecture comparison and length sweep**:

   ```bash
   python scripts/run_ck_sweep.py --seeds 0,1,2
   python scripts/run_length_sweep.py --seeds 0,1,2
   python src/compute_correlation.py --input sweep_results/architecture
   ```

The experiment scripts train real models with `scripts/train.py` and report measured values (validation loss, mode concentration, sample metrics, learned `tau` per secondary-structure class). Unknown options are forwarded to `train.py`, e.g. `--max_steps 20000 --d_model 384`. scTM is measured when `--proteinmpnn_dir` is given and is reported as null otherwise. Finished runs are skipped on restart.

## Notes on the Implementation

- The Rouse basis (DCT-II) and eigenvalues are computed for the actual chain length of each batch.
- The selective scan is evaluated in closed form in chunks (`scan_chunk`), which is equivalent to the step-by-step recurrence (`tests/test_core.py` checks this). `checkpoint_scan` recomputes it in the backward pass to save memory.
- The mode transform is a dense matrix product, O(L²) per layer in the chain length.
- The bond projection at the end of sampling makes every CA-CA distance exactly 3.8 Å. It does not constrain other geometry.
- Full-backbone PDB files are built from the CA trace as ideal planar trans peptides. The plane orientation is estimated from the CA trace (about 0.3 Å RMS error for N and C and 1.1 Å for O on crystal structures).
- Checkpoints carry a format version. A checkpoint written before an architecture change is rejected with an error and has to be retrained.

## License

MIT
