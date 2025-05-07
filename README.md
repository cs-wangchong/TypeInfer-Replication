# TIGER: A Generating-then-Ranking Framework for Practical Python Type Inference

This is the replication package for ICSE'25 paper "[TIGER: A Generating-then-Ranking Framework for Practical Python Type Inference](https://arxiv.org/pdf/2407.02095)"

## Source Code

The source files are located in the `typeinfer` directory:

* `model.py`: Defines the generation model and similarity model classes.
* `training.py`: Trains the generation and similarity models.
* `evaluation.py`: Loads trained models and runs evaluation on the test set.
* `compute_metrics.py`: Computes Top-1, Top-3, and Top-5 accuracy based on the predictions.

## Environment

* Python ≥ 3.9
* Required packages: `torch`, `transformers`, `hityper`

## Quick Reproduction

To quickly reproduce the results reported in our paper:

```bash
python typeinfer/compute_metrics.py
```

This script loads the predictions from `output/predictions/randomsampled.json` and computes evaluation metrics. Results and logs will be saved in `output/metrics/` and `output/logs/`, respectively.

## Full Reproduction

To fully reproduce the training and evaluation pipeline, follow the steps below.

### 1. Download Datasets

Download the processed datasets (data.zip) based on TypeGen from the released [resources](https://github.com/cs-wangchong/TypeInfer-Replication/releases/tag/v1.0).

### 2. Fine-tune Models

Train the generation and similarity models by running:

```bash
python typeinfer/training.py
```

Trained model checkpoints will be saved to the `models/` directory.

> **Note:** If you prefer not to retrain the models, you can download pre-trained checkpoints (models.zip) from the released [resources](https://github.com/cs-wangchong/TypeInfer-Replication/releases/tag/v1.0).

### 3. Run Evaluation

Run the evaluation on the test set:

```bash
python typeinfer/evaluation.py
```

Predictions will be saved to `output/predictions/randomsampled.json`.

Then compute the evaluation metrics:

```bash
python typeinfer/compute_metrics.py
```
