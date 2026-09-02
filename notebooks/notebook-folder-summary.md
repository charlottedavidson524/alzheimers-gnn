This folder is where all of the notebooks where the GNN models were trained will be contained. These were done in Google Colab to make use of the T4 GPU.

These are as follows:

- `notebooks/alzheimers_gnn.ipynb`: the first GNN
- `notebooks/alzheimers_gnn-tuning.ipynb`: the notebook containing hyperparameter tuning
- `notebooks/alzheimers_gnn-tuning_best.ipynb`: the notebook containing the GNN trained with the optimal hyperparameters.
- `notebooks/alzheimers_gnn_subject_level.ipynb`: notebook containing GNN after subject level aggregation
- `notebooks/alzheimers_gnn_subject_level_tuning.ipynb`: notebook containing hyperparameter tuning for GNN after subject level aggregation
- `notebooks/alzheimers_gnn_subject_level_BEST.ipynb`: Best outcome of hyperparameter tuning on GNN after subject level aggregation
- `notebooks/alzheimers-gnn-for-fusion.ipynb`: Retraining the GNN on the intersection of subjects shared with the tabular models for purposes of fusion.
