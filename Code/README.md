# Code Usage

This directory contains the LiTS example code for HMCI-Net.

Recommended entry points:

- `1_Train_LITS_main.py`: train HMCI-Net on a LiTS-style dataset.
- `Reference_LITS.py`: validate a trained checkpoint and save predictions.
- `networks/Heiix_Mamba/HM.py`: HMCI-Net model implementation.
- `load_datasets_transforms.py`: LiTS-only data loader and MONAI transforms used by the example scripts.

Run scripts from this directory so that relative imports resolve correctly:

```bash
cd Code
python 1_Train_LITS_main.py --root ../data/LITS/crop --network HM
```
