# Full-model fine-tuning results

Smartphone (PAD) validation balanced accuracy improved from 50.4% to 56.9%. This is validation performance used to choose the checkpoint, not independent test accuracy.

- mobilenet_finetune_v1: 52.9% PAD validation balanced accuracy, best epoch 5, fitting source combined, 16330 training images.
- mobilenet_pad_adapt_v1: 56.9% PAD validation balanced accuracy, best epoch 3, fitting source PAD-UFES-20, 1380 training images.

Selected model: mobilenet_pad_adapt_v1. SHA256: `a4619d1b9f73cdf2fa415e285642878472356cdd3a135dd8e6cd316d629db0d1`.

Combined warm-start fine-tuning: 5 epochs; PAD-only adaptation: 8 epochs. Both use all trainable layers, learning rate 0.00003, batch size 32, image size 160, seed 42, CPU, augmentation, and the same fixed original partitions. Checkpoint selection uses PAD validation. Original test images were not evaluated in either new run.

Historical baseline test results (49.6% combined; 52.6% PAD) apply only to the old frozen-feature model. They do not measure this model. External clinical evaluation and actual temperature calibration remain pending a new eligible dataset. Calibration tooling is implemented but no real-data temperature has been fitted. ISIC lesion grouping does not prove patient independence. This model remains an educational prototype.

DDI access requires individual registration and research-use terms at https://ddi-dataset.github.io/#access. It has not been downloaded or tested. Confirm label mapping and patient identifiers before defining separate calibration and independent evaluation cohorts. Do not substitute unsupported labels.
