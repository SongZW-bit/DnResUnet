# Computer Code Availability Template

This repository is prepared to support the `Computer Code Availability` requirement of *Computers & Geosciences*.

Recommended manuscript text:

> The source code used in this study is openly available at `REPOSITORY_URL` under the MIT License. The public release contains the realistic synthetic data generator (`forward_v2.py`), the configurable DnResUnet training framework (`DnResUnet_code.py`), the baseline training scripts, and the independent resampling evaluation workflow (`independent_resampling_eval.py`). A versioned archival snapshot of the release is available at `DOI_OR_ARCHIVE_LINK`.

Release checklist before publication:

- Replace `REPOSITORY_URL` with the final public repository link.
- Replace `DOI_OR_ARCHIVE_LINK` with the final archive DOI or permanent release URL.
- Confirm that `LICENSE` is present in the root of the repository.
- Confirm that the repository includes installation instructions and dependency specifications.
- Confirm that all checkpoints and sample files referenced by the paper exist in the tagged release.
- If the full synthetic dataset is archived separately, add a companion `Data Availability` statement in the manuscript.
