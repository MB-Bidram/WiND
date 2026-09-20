"""Versioned tensor checkpoints; no pickled model classes or tokenizer objects."""

from pathlib import Path
import os
import tempfile
import json

import torch


def save_checkpoint(path, payload):
    """Atomically replace a checkpoint on the same filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        record = {"format": "wind.language", "version": 1, **payload}
        if "config" in record:
            record["config_json"] = json.dumps(record["config"], sort_keys=True)
        torch.save(record, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format") != "wind.language" or data.get("version") != 1:
        raise ValueError("unsupported Wind language checkpoint")
    if "config" not in data or "config_json" not in data:
        raise ValueError("checkpoint is missing its embedded model config")
    try:
        if json.loads(data["config_json"]) != data["config"]:
            raise ValueError("embedded model config is inconsistent")
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid embedded model config") from exc
    return data


