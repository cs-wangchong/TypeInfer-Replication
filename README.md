# TIGER - Replication Package

The source code and prediction results of our approach are in `typeinfer` and `output/predictions/randomsampled.json`, respectively. 

##
To reproduce the results reported in our paper, please directly run `python typeinfer/compute_metrics.py`, the results will be saved into `output/metrics` and `output/logs`.


## 
To achieve full reproduction, please execute the following steps.

### Download datasets


### Fine-tune models
run `python typeinfer/training.py`

The resulting models will be saved into a new foloder named `models`.

### Run evaluation
config the paths of the resulting models at line 9 and line 10.

run `python typeinfer/evaluation.py`

The prediction results will be saved as `output/predictions/randomsampled.json`.

run `python typeinfer/compute_metrics.py`
