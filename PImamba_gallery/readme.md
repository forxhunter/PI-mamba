Six PI-Mamba backbones of length 100, taken from `PImamba_L100/`.

| File | Source | scTM |
|---|---|---|
| sample_1.pdb | PM_030 | 0.972 |
| sample_2.pdb | PM_070 | 0.962 |
| sample_3.pdb | PM_000 | 0.961 |
| sample_4.pdb | PM_060 | 0.954 |
| sample_5.pdb | PM_040 | 0.950 |
| sample_6.pdb | PM_080 | 0.947 |

scTM was measured with `scripts/evaluate.py --mode eval` (CA-only ProteinMPNN, 4 sequences per backbone at temperature 0.1, ESMFold).

The CA coordinates are those of the source files. N, C and O were placed around the CA trace with `src.geometry.ca_to_backbone` (ideal planar peptides, plane orientation estimated from the CA trace). `sample_gallery.png` is a PyMOL cartoon rendering of the six files.

These files replace an earlier set whose coordinates did not form a connected chain (mean CA-CA distance 8.7 A).
