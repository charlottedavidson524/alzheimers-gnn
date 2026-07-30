"""
This is for environment-aware path resolution for Colab and local Windows.

Will allow me to train in Colab using GPUs
"""

from pathlib import Path

def get_graphs_root() -> Path:
    """
    Return the graphs directory, detecting the environment.
    """
    candidates = [
        Path("/content/graphs"), # Colab local 
        Path("/content/drive/MyDrive/alzheimers-gnn-data/graphs"), # Colab Drive
        Path("C:/data/pearl-neuro/derivatives/graphs"),  # Windows local
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No graphs directory found.")


def get_participants_tsv() -> Path:
    """
    Return the participants.tsv file, detecting the environment
    """
    candidates = [
        Path("/content/participants.tsv"),
        Path("/content/drive/MyDrive/alzheimers-gnn-data/participants.tsv"),
        Path("C:/data/pearl-neuro/participants.tsv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No participants.tsv found.")


def get_results_root() -> Path:
    """
    Return where to save results, detecting the env.
    """
    if Path("/content/drive/MyDrive/alzheimers-gnn-data").exists():
        results = Path("/content/drive/MyDrive/alzheimers-gnn-data/results")
        results.mkdir(exist_ok=True)
        return results
    return Path("C:/dev/alzheimers-gnn/results")