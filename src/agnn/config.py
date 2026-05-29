from pathlib import Path
import yaml

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_config(path: str = "config/default.yaml") -> dict:
    """Load the project config as a nested dictionary"""
    config_path = PROJECT_ROOT / path
    with open(config_path, "r", encoding="utf=8") as f:
        return yaml.safe_load(f)