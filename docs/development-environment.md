# Development Environment

Project uses a hybrid workflow. VSCode on Windows for development, Google Colab for GPU training, GitHub as the bridge, and Google Drive for data and results

All code is written in VSCode on Windows. Colab is never used for editing code. Any changes made there are lost when the session ends. Push to GitHub. Pull in Colab.

Guide to where code is run:

| Task                        | Environment  | Reason                                                    |
| --------------------------- | ------------ | --------------------------------------------------------- |
| EEG preprocessing pipeline  | VSCode (CPU) | Already complete                                          |
| Graph construction pipeline | VSCode (CPU) | Already complete                                          |
| Skeleton tests (5 subjects) | VSCode (CPU) | Fast checking of code changes                             |
| Full GNN training           | Colab (GPU)  | Would take hours on CPU per run but likely minutes on GPU |
| Cross-validation            | Colab (GPU)  | Too slow on CPU                                           |
| Hyperparameter sweeps       | Colab (GPU)  | Only possible with GPU                                    |
| Fusion experiments          | Colab (GPU)  | Multi-modal training                                      |

A typical session might look like this:

In VSCode:

- Make code changes
- Test locally on skeleton subjects
- Commit and push to GitHub

In Colab:

- Run the setup cell that has been created (mount Drive, `git pull`, copy data to local storage)
- Restart runtime if code changed
- Run training
- Save results to Drive

Trained models and metrics live in `Google Drive/alzheimers-gnn-data/results/`. Nothing important is saved to `/content/` because that folder gets wiped when Colab sessions end

The `src/agnn/paths.py` module detects the environment automatically. Any code that needs a data path calls:

```python
from agnn.paths import get_graphs_root, get_participants_tsv, get_results_root
```

Returns Windows paths locally, Colab paths in Colab. No code changes needed between environments.
