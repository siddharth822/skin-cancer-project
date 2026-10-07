# Current development status

Full-model fine-tuning and PAD adaptation are complete. The selected model is connected to the app. See [fine-tuning results](reports/finetune_v1/RESULTS.md). PAD validation balanced accuracy is 56.9%, compared with 50.4% for the warm-start baseline on the same validation cohort. No new independent-test accuracy is established.

Eighteen regression tests passed. External-cohort overlap auditing, evaluation and temperature calibration commands are implemented. Actual external clinical evaluation and calibration still require a new eligible labeled dataset. The old test partition is excluded from the new training runs.

The checkpoint is models/skinsight_mobilenetv3_small.pt, SHA256 a4619d1b9f73cdf2fa415e285642878472356cdd3a135dd8e6cd316d629db0d1. Code and aggregate reports are stored in Git; weights are included in the downloadable trained app. Live processes must restart in future tasks. Fresh-task restoration has not been independently verified.
