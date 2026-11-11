"""Save and load trained CrossRhythm models.

A checkpoint is a dictionary with the identity of the run (dataset, subject,
seed), the model dimensions, the network weights and the fitted preprocessing
(EA matrix and standardization of the target participant, filter settings and
the identifiers of the calibration trials).
"""

from pathlib import Path

import torch

from .model import CrossRhythm


def _to_tensors(preprocessing):
    state = dict(preprocessing)
    state["alignment"] = {key: torch.as_tensor(value, dtype=torch.float64)
                          for key, value in preprocessing["alignment"].items()}
    state["normalization"] = {key: {name: torch.as_tensor(value) for name, value in moments.items()}
                              for key, moments in preprocessing["normalization"].items()}
    return state


def _to_numpy(preprocessing):
    state = dict(preprocessing)
    state["alignment"] = {key: value.numpy() for key, value in preprocessing["alignment"].items()}
    state["normalization"] = {key: {name: value.numpy() for name, value in moments.items()}
                              for key, moments in preprocessing["normalization"].items()}
    return state


def save_checkpoint(path, model, model_config, preprocessing, dataset, subject, seed):
    """Store the weights and only the target participant's preprocessing statistics."""
    target = str(int(subject))
    kept = dict(preprocessing)
    kept["alignment"] = {key: value for key, value in preprocessing["alignment"].items() if key == target}
    if preprocessing["normalization_scope"] == "source_and_target":
        kept["normalization"] = {"target": preprocessing["normalization"]["target"]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dataset": dataset, "subject": int(subject), "seed": int(seed), "model": model_config,
                "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "preprocessing": _to_tensors(kept)}, path)


def load_checkpoint(path, device="cpu"):
    """Return (model in evaluation mode, preprocessing state, checkpoint dictionary)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(checkpoint["model"])
    config.pop("name", None)
    model = CrossRhythm(**config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, _to_numpy(checkpoint["preprocessing"]), checkpoint
