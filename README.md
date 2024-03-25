# TIGER - Replication Package

## Source Code
The source code is in the folder named `typeinfer`:
- model.py: Define the classes of generation model and similarity model
- training.py: Train the generation model and similarity model
- evaluation.py: Load trained models and run the evaluation
- compute_metrics.py: Compute Top-1,3,5 accurary based on the predictions

## Evironment
- python >= 3.9
- packages: torch, transformers, hityper

## Quick Reproduction
To reproduce the results reported in our paper, please directly run `python typeinfer/compute_metrics.py`. This will load the predictions we have saved at `output/predictions/randomsampled.json`, and output the metrics and intermediate information into `output/metrics` and `output/logs`, respecitively.

## Full Reproduction
To perform the full reproduction including model training and testing, please execute the following steps.

### Download datasets
Download the datasets from TypeGen's replication package using this [link](https://github.com/JohnnyPeng18/TypeGen/releases/tag/data) and put the json files into the folder `data`.

### Fine-tune models
run `python typeinfer/training.py` to train the generation model and similarity model. The resulting models will be saved into a new foloder named `models`.

**Note:** If you want to train the models, you can download our checkpoints shared in figshare: [link](https://figshare.com/s/927f2337505a7ea66ce1).

#### Run evaluation
Run `python typeinfer/evaluation.py` to perform the evaluation on the testset. The prediction results will be saved as `output/predictions/randomsampled.json`.

Run `python typeinfer/compute_metrics.py` to compute accuray.
