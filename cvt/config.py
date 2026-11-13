"""Config file loading."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return data


def load_models(spec: str | Path) -> dict:
    """Detector registry from comma-separated yaml files (e.g. `configs/models.yaml,
    configs/models_deploy.yaml`), merged in order; a later file may not redefine a name."""
    out: dict = {}
    where: dict[str, str] = {}
    files = [s.strip() for s in str(spec).split(",") if s.strip()]
    if not files:
        raise ValueError(f"no models file given ({spec!r})")
    for path in files:
        for name, entry in load_yaml(path).items():
            if name in out:
                raise ValueError(f"{path}: model {name!r} is already defined in {where[name]}")
            out[name], where[name] = entry, path
    return out


def read_exclusions(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return {line for line in lines if line and not line.startswith("#")}
