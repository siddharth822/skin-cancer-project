# Real-data baseline results

A five-epoch frozen-feature MobileNetV3-Small baseline was trained on the complete
six-class dataset (27,137 matched images), with no train-time augmentation.
The backbone remained at pretrained ImageNet weights; only the classifier was fitted.
Epoch 4 was selected by validation balanced accuracy.
The fixed test partition was evaluated after checkpoint selection, without tuning
against test results. This is a research prototype, not a reliable medical classifier.

| Held-out source | Images | Balanced accuracy |
|---|---:|---:|
| Combined | 5335 | 49.6% |
| PAD smartphone photographs | 436 | 52.6% |
| ISIC dermoscopy | 4899 | 47.4% |

Balanced accuracy is the mean of class sensitivities, not ordinary accuracy.
The PAD melanoma test contains only 5 images: 3/5 were correctly classified.
That sample is far too small for dependable melanoma performance claims.
Scores are uncalibrated; external smartphone accuracy remains unmeasured.
Official ISIC metadata enables lesion grouping where present, not guaranteed patient
independence. Linked identities and exact-file hashes were verified disjoint across
train/validation/test. Re-encoded or cropped duplicates are not ruled out.

The actual checkpoint loaded successfully in the app. A held-out PAD image returned
HTTP 200 with the expected prediction, guidance, and staging limitations. The home
page and health endpoint passed. Thirteen software regression tests passed.
The checkpoint is intentionally excluded from Git and supplied in the downloadable
trained-app ZIP and prepared environment filesystem.

Training source commit: `63a917f782d16354265c4ce1205d0f8630fd2495`.
Checkpoint SHA-256: `e8d79d8600324d90e96a37f55163c653060b23691608a907e4c12dc5b1c6b5fb`.
Split summary and complete aggregate validation/test reports accompany this file.
No individual patient records or test image paths are included in Git reports.

Next research steps: fine-tune using validation only, obtain genuine ISIC patient
mapping where possible, establish a new independent external test, and assess
calibration. Do not tune further using this already-reported test partition.
