from pathlib import Path


def get_config_path(name: str) -> Path:
    """Return the path to a YAML config shipped with the package."""
    return Path(__file__).resolve().parent / name
