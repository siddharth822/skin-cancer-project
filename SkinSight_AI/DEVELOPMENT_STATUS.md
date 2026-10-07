# Current development status

The real-data baseline is trained and connected to the app. See
[baseline results](reports/baseline_v1/RESULTS.md) for measured performance and limits.
Thirteen regression tests passed. Downloads succeeded after network publication;
the parallel PAD transfer passed the publisher-provided MD5 checksum check.
The home page, health endpoint and a real held-out PAD image prediction passed.

The baseline uses frozen pretrained MobileNet features and five classifier epochs,
not full-model fine-tuning. Combined held-out balanced accuracy is 49.6%; PAD-only
balanced accuracy is 52.6%. These results do not support medical deployment.
Genuine ISIC patient mapping, external validation and calibrated probabilities remain
outstanding. Do not reuse the reported test partition for model tuning.

The prepared environment contains the checkpoint at
models/skinsight_mobilenetv3_small.pt. Git stores code and aggregate reports;
downloadable trained-app ZIP stores the checkpoint. Live processes must restart
in future tasks. A fresh-task restoration has not been independently verified.
