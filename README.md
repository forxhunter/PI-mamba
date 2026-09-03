# PI-Mamba: Physics-Informed State Space Models for Protein Design

This repository contains the official implementation of **PI-Mamba**, corresponding to the paper "PI-Mamba: Scalable Protein Backbone Generation via Physics-Informed Spectral Initialization".

+ We are aware there are some scripts not the latest version, which will be released soon!
## Repository Structure

- `src/`: Core model implementation
  - `backbone_pi_mamba.py`: Main PI-Mamba architecture
  - `pi_mamba_layer.py`: Bidirectional Mamba block with Rouse initialization
  - `rouse_physics.py`: Spectral initialization logic
  - `geometry.py`: NeRF and coordinate utilities
- `scripts/`: various training and sweeping scripts
  - `train.py`: Main training script *(not yet released — see note under Training)*
  - `validate_physics.py`: Physics validation (Figure 4)
  - `run_counterfactual_experiments.py`: Ablation studies (Appendix S16)
- `configs/`: YAML configuration files (reproduction)
- `PImamba_gallery/`: PI-mamba 6 protein shown in the paper as gallery.
- `PImamba_L100/`: PI-mamba 100 proteins with Length=100 and scTM~ 0.91.
- `Distilled_trainset/`: Distilled training set with scTM >= 0.95 from Proteina with more than 6000+ pdb structures for training.

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**

- PyTorch 2.1+
- Mamba-SSM (`pip install mamba-ssm`)
- Einops, Hydra-Core, Biopython

## Dataset

This codebase expects PDB files (e.g., CATH 4.2 dataset).
Point the `--data_dir` argument to your folder containing `.pdb` files.

## Training

> **Note:** `scripts/train.py` is not part of this release yet. The commands below
> document its intended interface and will work once the training script is published.

To train the full PI-Mamba model (Paper Config: L=16, D=512):

```bash
python scripts/train.py --use_physics True --data_dir /path/to/pdb/files
```

To train the ablation baseline (Learned-A):

```bash
python scripts/train.py --use_physics False
```

## Reproducing Results

1. **Physics Validation** (Figure 4):

   ```bash
   python scripts/validate_physics.py
   ```

2. **Counterfactual Experiments** (Appendix S16):
   ```bash
   python scripts/run_counterfactual_experiments.py
   ```

## License

MIT
