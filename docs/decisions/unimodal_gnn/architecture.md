# Architecture (Primary GNN)

In summation, this model is a two branch, band parallel GCN. There is one small GCN for each frequency band (delta and aplha-2), each trained independently, combined by concatentaion after graph concolution followed by a small classifier.

Per branch design:

- Input graph: 125 electrode nodes, edges = wPLI connectivity for that band, top 20% proportional threshold.
- Node features: each node has 1 value. This is the node's relative power in that branch's own band (delta branch → delta power; alpha-2 branch → alpha-2 power). There's no cross-band info in a branch's node features.
- Layers are as follows: 2-layer GCN, each layer -> ReLU -> batch norm
- Readout: max pooling over all nodes -> one embedding vector per branch.
- Weights: each branch trains its own weights, not shared with the other branch.

The fusion strategy:

- The 2 branch embeddings are concatenated after graph convolution completes.
- No cross band interaction happens during message passing. It only happens at this point.

Classifier head:

- There are 2–3 fully connected layers on the concatenated vector with dropout between them.
- The output should be 2 units (carrier/non-carrier), softmax and cross-entropy loss.

Data and evaluation:

- Per-epoch graphs should be used as separate training examples (not one graph per subject).
- Subject-level grouped k-fold CV —> no subject's epochs can appear in both train and test.
- Per-epoch predictions are aggregated to one subject-level prediction at evaluation (majority vote or averaged probability).
