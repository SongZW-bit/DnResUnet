# Sample Data Notes

This folder contains:

- `gravity_dataset_v2_smoketest.pt`
  A very small smoke-test dataset for checking that the environment, tensor loading, and plotting scripts work correctly.
- `gravity_dataset_v2_smoketest.json`
  Metadata corresponding to the smoke-test dataset.
- `gravity_dataset_100k_V2_REALISTIC.json`
  Metadata for the full realistic synthetic dataset used in the manuscript.

The full `gravity_dataset_100k_V2_REALISTIC.pt` training corpus is not bundled here by default because large binary datasets are better archived separately from the source-code repository. It can be regenerated with:

```bash
python forward_v2.py --samples-per-level 20000 --output gravity_dataset_100k_V2_REALISTIC.pt
```
