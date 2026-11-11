"""Evaluation on the later sessions in the static and the causal setting.

Static: the alignment, standardization and batch-normalization (BN) statistics
estimated from the calibration data are kept fixed for the whole test session.

Causal: the trials are processed one at a time in recording order. Each trial is
first classified, and only afterwards are the statistics updated with it:

* EA reference (only for decoders that use EA): running mean of the uncentered
  covariances, initialized with the N_c calibration trials.
* BN statistics: initialized with the mean and variance of the calibration
  trials and updated as a cumulative mean with weight 1 / (N_c + t).

Labels and future trials are never used, input standardization stays fixed, and
every test session starts again from the calibration statistics.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from .preprocessing import apply_preprocessing, filter_epochs, inverse_sqrt, mean_covariance


def recording_order(rows):
    """Positions of the trials sorted by participant, session, run and onset."""
    runs = rows["run"].astype(str)
    order = pd.DataFrame({
        "subject": rows.subject.astype(int).to_numpy(), "session": rows.session.astype(int).to_numpy(),
        "run": pd.to_numeric(runs, errors="coerce").fillna(runs.map({"A": 0, "B": 1})).to_numpy(),
        "onset": rows.raw_onset_sample.to_numpy(), "index": rows.array_index.to_numpy(),
        "trial_id": rows.trial_id.astype(str).to_numpy(), "position": np.arange(len(rows))})
    return order.sort_values(["subject", "session", "run", "onset", "index", "trial_id"],
                             kind="stable").position.to_numpy()


@torch.no_grad()
def predict_static(model, test, preprocessing, device="cpu", batch_size=64):
    """Class probabilities with fixed calibration statistics."""
    model.eval()
    x = apply_preprocessing(test["X"], test["trials"].subject, preprocessing)
    probabilities = []
    for start in range(0, len(x), batch_size):
        batch = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
        probabilities.append(model(batch).softmax(1).cpu().numpy())
    return np.concatenate(probabilities)


class _BatchNormTracker:
    """Collects the input moments of every BN layer during a forward pass."""

    def __init__(self, model):
        self.layers = {name: layer for name, layer in model.named_modules()
                       if isinstance(layer, nn.modules.batchnorm._BatchNorm)}
        self.pending = {}
        self.handles = [layer.register_forward_pre_hook(self._hook(name))
                        for name, layer in self.layers.items()]

    def _hook(self, name):
        def collect(layer, inputs):
            x = inputs[0].detach().double()
            axes = (0, *range(2, x.ndim))
            count = x.numel() // x.shape[1]
            variance, mean = torch.var_mean(x, dim=axes, unbiased=True)
            m2 = variance * (count - 1)
            if name in self.pending:  # merge if a layer is used more than once
                old_count, old_mean, old_m2 = self.pending[name]
                total = old_count + count
                difference = mean - old_mean
                m2 = old_m2 + m2 + difference.square() * (old_count * count / total)
                mean = old_mean + difference * (count / total)
                count = total
            self.pending[name] = (count, mean, m2)
        return collect

    def snapshot(self):
        return {name: (layer.running_mean.clone(), layer.running_var.clone(),
                       layer.num_batches_tracked.clone()) for name, layer in self.layers.items()}

    @torch.no_grad()
    def restore(self, snapshot):
        for name, layer in self.layers.items():
            mean, variance, count = snapshot[name]
            layer.running_mean.copy_(mean)
            layer.running_var.copy_(variance)
            layer.num_batches_tracked.copy_(count)

    @torch.no_grad()
    def update(self, seen):
        """Cumulative-mean update of every BN layer with the moments of the last trial."""
        for name, (count, mean, m2) in self.pending.items():
            seen[name] += 1
            weight = 1 / seen[name]
            layer = self.layers[name]
            layer.running_mean.lerp_(mean.to(layer.running_mean.dtype), weight)
            layer.running_var.lerp_((m2 / (count - 1)).to(layer.running_var.dtype), weight)
            layer.num_batches_tracked.add_(1)

    def remove(self):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def predict_causal(model, test, calibration, preprocessing, device="cpu"):
    """Class probabilities with predict-then-update adaptation (see module docstring).

    `calibration` holds the EEG of the target's calibration trials (no labels).
    """
    model.eval()
    target = int(preprocessing["target_subject"])
    tracker = _BatchNormTracker(model)
    original = tracker.snapshot()
    try:
        # Initial BN statistics: mean of the per-trial moments of the calibration trials.
        calibration_x = apply_preprocessing(calibration["X"], calibration["trials"].subject, preprocessing)
        sums = {name: [0.0, 0.0] for name in tracker.layers}
        for trial in calibration_x:
            tracker.pending.clear()
            model(torch.as_tensor(trial[None], dtype=torch.float32, device=device))
            for name, (count, mean, m2) in tracker.pending.items():
                sums[name][0] = sums[name][0] + mean
                sums[name][1] = sums[name][1] + m2 / (count - 1)
        n_calibration = len(calibration_x)
        for name, layer in tracker.layers.items():
            layer.running_mean.copy_((sums[name][0] / n_calibration).to(layer.running_mean.dtype))
            layer.running_var.copy_((sums[name][1] / n_calibration).to(layer.running_var.dtype))
            layer.num_batches_tracked.fill_(n_calibration)
        initial_bn = tracker.snapshot()

        # Initial EA reference: mean covariance of the calibration trials (filtered, not aligned).
        uses_ea = str(target) in preprocessing["alignment"]
        if uses_ea:
            filtered_calibration = filter_epochs(calibration["X"], preprocessing["filter"])
            initial_reference = mean_covariance(filtered_calibration)

        probabilities = [None] * len(test["X"])
        current_session = None
        for position in recording_order(test["trials"]):
            row = test["trials"].iloc[position]
            session = (int(row.subject), int(row.session))
            if session != current_session:  # every session starts from the calibration state
                tracker.restore(initial_bn)
                seen = {name: n_calibration for name in tracker.layers}
                if uses_ea:
                    reference, n_reference = initial_reference.copy(), n_calibration
                    alignment = preprocessing["alignment"][str(target)]  # = inverse_sqrt(reference)
                else:
                    alignment = None
                current_session = session

            tracker.pending.clear()
            raw = test["X"][position:position + 1]
            x = apply_preprocessing(raw, [target], preprocessing, alignment=alignment)
            logits = model(torch.as_tensor(x, dtype=torch.float32, device=device))
            probabilities[position] = logits.softmax(1)[0].cpu().numpy()

            # Update only after the prediction of this trial has been made.
            if uses_ea:
                filtered = filter_epochs(raw, preprocessing["filter"])[0]
                reference = (n_reference * reference + filtered @ filtered.T) / (n_reference + 1)
                n_reference += 1
                alignment = inverse_sqrt(reference)
            tracker.update(seen)
        return np.stack(probabilities)
    finally:
        tracker.remove()
        tracker.restore(original)
