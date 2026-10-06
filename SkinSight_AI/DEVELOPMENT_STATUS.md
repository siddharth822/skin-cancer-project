# Verified development status

The app and trainer now share torchvision MobileNetV3-Small and the same checkpoint
filename. The predictor uses a restricted checkpoint loader. Static assets and the
default checkpoint resolve relative to the application files rather than the shell.

Dataset preparation validates one-hot ISIC labels and image matching, preserves
patient/lesion grouping where metadata exists, and hashes image files. Training
reserves train/validation/test folds, checks source/class coverage, selects weights
on validation only, and exports overall and source-specific held-out reports.

## Verification completed

- Twelve regression tests: HTTP prediction, invalid image and missing model responses,
  checkpoint loading, recursive manifests, one-hot validation, dummy-ID handling,
  transitive identity grouping, partition isolation, and metric calculations.
- Two epochs on 240 synthetic images: training, best-checkpoint selection, test
  isolation, report generation, checkpoint reload, and HTTP prediction.
- Uploaded metadata: 2,298 PAD records and 24,839 supported ISIC records permit
  source/class coverage in three partitions. This metadata-only feasibility check
  did not include image hashes or the official ISIC lesion metadata; final folds
  must be generated from actual downloaded images and metadata.
- Dependencies passed pip check. Notebook code cells passed syntax compilation;
  exported source ZIP passed integrity checks and contains no synthetic checkpoint.

## Outstanding external work

No real-image model has been trained in this workspace. Kaggle and official ISIC
HTTP requests were denied by the cloud egress proxy. GPU hardware and Kaggle runtime
access are unavailable here. Network domain additions were saved to the environment
draft but were not applied to the live instance during validation.

Full downloads and GPU notebook execution remain unverified. Research accuracy,
external smartphone performance, genuine ISIC patient separation, and probability
calibration cannot be inferred from software tests. The notebook includes a fixed
internal test; repeated tuning against its results invalidates independence.

The next supported step is to apply the saved network settings and retry cloud
access, or execute the bundled notebook in a Kaggle account with GPU/Internet access.
Only deploy the resulting real-data checkpoint after reviewing the test report.
