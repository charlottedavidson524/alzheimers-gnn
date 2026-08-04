# First Pass

This is the document where I will list all of the decisions I made when building the first GNN.

## Architecture:

I decided to use a two branch Graph Convolutional Network (GCN) with parallel per-band branches, concatentaion, and an MLP classifier for APOE e4 binary classification.

The architecture is as follows:

Delta graph -> GCN branch (2 layers, 64 -> 32) -> 32-dim embedding

Alpha-2 graph -> GCN branch (2 layers, 64 -> 32) -> 32-dim embedding

Concat (64) -> Linear(64 -> 32) -> ReLU -> Dropout -> Linear(32 -> 2)

There are roughly 6300 trainable parameters.

A few alternatives that colud be considered included:

- Single band GCN (alpha-2 only or delta-2 only)
- Six band GCN (this was my original plan)
- Graph Attention Networks (GAT)
- Adaptive Gated GCN (AGGCN, Klepl et al., 2023)
- Multi-dimensional edge features in a single graph
- Deeper GCN (3 or more layers)

Justification for these architectural decisions:

- Two branches match the two band frequency decision: the functional connectivity plan specifies two frequency bands (delta 0.5-4 Hz, alpha-2 10-12 Hz). Each abnd gets its own graph per epoch, and and each graph needs its own GCN branch.

- GCNConv matches the standard-GCN scalar-edge constraint: Kipf & Welling's (2017) GCNConv naturally consumes scalar edge weights using the `edge_weight` argument. This matches the decision to use scalar wPLI as edge features rather than multi-dimensional vectors, which would mean using a different convolution operator.

- 2 layers matches Klepl et al.'s proven configuration: Klepl et al. (2022, 2023) use 2 GCN layers as their standard configuration for AD-EEG-GNN classification. Deeper networks introduce oversmoothing at the graph scale (125 nodes) without adding useful capacity. Klepl's later work with attention still uses 2 GCN layers as the base.

- The parameter count fits N=77: with approximately 6,300 trainable parameters (across both branches and the classifier head), the model has approx 80 params for each training subject. This should be within the range where reliable training is possible without severe overfitting. A six-branch equivalent would triple parameters without dual-precedent justification for the extra bands.

- Per-branch node feature isolation prevents information leakage: each branch receives only its own band's node feature (1-dim per node). Delta branch sees delta relative power and alpha-2 branch sees alpha-2 relative power. This keeps branches self-contained regarding their information. Cross-band interactions are learned by the MLP classifier from the concatenated embeddings, not by the GCN branches themselves.

- Concatenation-before-classifier matches Xu et al. and Klepl et al: the pattern of running per-band branches in parallel with each other concatenating their embeddings before a classifier head is used in the lit (Xu et al. 2025, Klepl et al. 2023). Lets each branch specialise in its band's connectivity structure while the classifier learns to combine information across bands.

Layer choices:

- GCN layer type: `GCNConv` (Kipf & Welling 2017). Supports scalar edge weights natively. Well tested in PyTorch Geometric.

- Hidden dimension: 64. This matches Klepl et al.'s hidden dim for AD-EEG-GNN work. Small enough to prevent overfitting at N=77 but shpould be large enough to capture per band coupling structure. Can drop to 32 if hyperparameter tuning indicates overfitting.

- Embedding dimension: 32: each branch produces a 32-dim graph-level embedding. Concatenating two branches gives a 64-dim vector for the classifier which matches the classifier's input dimension.

- Activation function: ReLU. This is the standard choice for GCNs. Matches Klepl et al. Easier and faster than alternatives (ELU, LeakyReLU) with no evidence of substantial improvement for AD-EEG-GNN tasks.

- Graph-level pooling: `global_mean_pool`. This averages node embeddings across each graph to produce a graph-level embedding. Standard for graph classification tasks. Alternatives (max pooling, sum pooling) are more sensitive to graph size and outlier nodes whereas mean pooling is robust to both.

- Dropout: 0.5, applied between GCN layers and in classifier head. This regularises against overfitting at small N. Standard rate for GNNs. Applied at two locations. Firstly, after the first GCN layer's ReLU (regularising node features) and then inside the MLP classifier (regularising the graph-level representation). Can configure as a hyperparameter.

- Classifier head: 2-layer MLP (64 -> 32 -> 2): This is more expressive than a single linear layer while adding minimal parameters. lETS the classifier learn non linear combinations of the two branch embeddings.

Didnt use AGGCN: Klepl et al. (2023) introduced adaptive gated GCN (AGGCN) for AD classification, which uses attention over edges. the advantages are tghat it has better interpretability of which connections drive the prediction. Thie disadvantages are that there are more parameters, it's more complex to implement and debug, and Klepl's own results show only modest improvements over vanilla GCN. It makes esnse to stick with vanilla GCN as the first pass. If baseline performance is poor or if edge-level interpretability becomes a priority, can try AGGCN.

Also didn't use GAT: Graph Attention Networks (GAT) learn attention weights over neighbours. This could (in theory) capture more nuanced relationships. Disadvantages include more parameters, the fact that attention doesn't directly consume edge weights the way GCNConv does and so requires architectural workarounds and the fact that there's no strong precedent in the AD-EEG-GNN literature. Vanilla GCN is easier to implement and has better precedent.

Could've used a deeper network but didn't. Adding more GCN layers (3, 4, or more) creates the oversmoothing problem: node embeddings become increasingly similar to each other which reduces discriminative power. For 125-node graphs, 2 GCN layers propagates information across 2-hop neighbourhoods, which should be enough to cpature local connectivity structure. If deeper propagation is needed later on then GCN2Conv with residual connections or GraphSAGE are potential alternatives.

When implementing there are two classes in `src/agnn/gnn/model.py`:

- `BandBranch`: this is one 2-layer GCN branch. Its a resuable building block and can instantiate one per band.

- `MultiBandGCN`: this is the full model with 2 branches and the MLP classifier.

Forward signature: `model(delta_batch, alpha2_batch)` where each argument is a `torch_geometric.data.Batch` of that band's graphs. Returns logits of shape `(batch_size, 2)`.

Node feature slicing per branch is done inside the model's forward pass — each graph's `x` has both band powers (columns 0 and 1), but each branch only accesses its own column.

The architectuire needs adjusting if:

- Baseline performance is poor (AUC < 0.65). Consider GAT or AGGCN.
- Overfitting is severe (train AUC >> validation AUC). Then can try reducing hidden dimension to 32 and increasing dropout to 0.6-0.7.
- Underfitting is severe (train AUC ~ 0.5): increase hidden dimension to 128 and/or add a third GCN layer.
- Interpretability of specific edges is needed.Can switch to AGGCN.
- Additional bands are added.Easy, instantiate more `BandBranch` objects and increase the classifier input dim.

## Loading and Cross-Validation

Training samples are structured as pairs. They look like (delta_graph, alpha2_graph, label). Cross-validation uses `StratifiedGroupKFold` with subjects as groups and APOE labels as the stratification target.

Each training sample is a tuple of two graphs from the same (subject, epoch), plus a single shared APOE label. The delta and alpha-2 graphs are paired at load time and go through the pipeline together. This is becayse it matches the two-branch model architecture. The model's forward signature is `model(delta_batch, alpha2_batch)`, which requires the data loader to output aligned pairs.

Cross-validation is done using `StratifiedGroupKFold`. Splits are made once at the start of training and reused across all folds. This is the configuration:

- Groups: subject IDs. No subject can appear in both train and test for any fold. TYhis is enforced by the splitter and verified explicitly at fold start using the `verify_no_subject_leakage` function.
- Stratification: binary APOE labels. Each fold has similar carrier/non-carrier proportions. This is to avoid pathological folds (e.g. all-carrier test set).
- 5 folds, shuffle=True, seed=42. This is standard for reproducible k-fold.

The reasoing for this is that Brookshire et al. (2024) showed that segment-level splitting on Alzheimer's EEG classifiers inflates test-set accuracy from around 50%/chance to 99.8%. Subject-level splitting is the main defence against this leakage. Stratification is a secondary requirement. It ensures meaningful metrics per fold given the mild class imbalance (1.5:1 carriers:non-carriers).

For batching there is a custom collation function. `collate_pairs` produces two aligned `torch_geometric.data.Batch` objects plus a label tensor. The i'th delta graph in `delta_batch` corresponds to the i'th alpha-2 graph in `alpha2_batch`. This is because the standard PyG `DataLoader` batches Data objects individually. There's no built-in way to batch pairs of Data that I'm aware of.

I have included defensive leakage checks. `verify_no_subject_leakage` is called at the start of each fold to clearly verify no subject overlaps between train and test indices. This isn't necessarily needed given the `StratifiedGroupKFold` splitter's guarantees, but I'm hoping it can be defensive against silent bugs (splitter misuse, index manipulation errors, upstream refactors).

Will adjust the data loading approach if:

- Class imbalance shifts substantially (e.g. beyond 3:1). Might then consider oversampling the minority class
- Memory becomes a bottleneck. There is currently roughly 13,500 pairs x 2 graphs held in memory (~1-2 GB). Might need on-demand loading if scaled up substantially.
- Sample-level augmentation is introduced. Would need to include per-batch augmentation logic in the collate function.

## Training

Can find the training module under `src/agnn/gnn/train.py`. This file provides reusable functions for training a single fold of the 2 branch GNN. This is done to separate the traininbg mechanics (loop, loss, optimiser, early stopping and checkpointing) from the driver logic like cross-validation, cohort iteration and results aggregation so they can both be tested and modified independently. This is good for testability and reusability. There are four functions that each have their own responsibility:

- `set_seed(seed)` -> Call this at the start of each training run and per fold. This makes sure that cross validation results can be compared across runs and hyperparameter configurations.

- `train_one_epoch(model, train_loader, optimizer, criterion, device)` -> this runs one epoch of forward-backward-step over the training loader. It returns the average training loss weighted by btach size. It's isolated from the train_fold for testability and reusability purposes.

- `evaluate(model, loader, criterion, device)` -> This runs the model in evaluation mode over a data loader. Returns a dictionary with average loss, ROC-AUC and per sample predictions.

- `train_fold(model, train_loader, val_loader, config, device, checkpoint_path, verbose)` ->this is the main entry point and trains one fold to converghence with class weighted cross entropy loss (handles 1.5:1 class imbalance), adam optimizer with weight decay, early stopping on validation AUC and model checkpointing to save state of best AUC model to disk. This returns a dictionary with various info and performance metrics.

This all separates the mechanics from the experiments which are done in Google Colab to use T4 GPU. These colab experiments will be saved to a folder called `notebooks`.

## Results

See the file under `notebooks/alzheimers-gnn.ipynb` which contains a markdown cell at the bottom describing the results of the first GNN. Any hyperparameter tuning or further attempts at improvement of the GNN will also be recorded in notebooks under this folder.
