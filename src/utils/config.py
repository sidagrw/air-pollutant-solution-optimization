"""Configuration utilities."""

import yaml


def load_yaml(path: str):
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
