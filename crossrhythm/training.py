"""Two-stage training of CrossRhythm.

Stage 1 (pretraining) uses the source set and stage 2 (calibration) the target's
calibration set. Both stages use AdamW, a linear warm-up followed by cosine
annealing, label smoothing and an exponential moving average (EMA) of the
weights. The EMA weights at the end of a stage are used without validation-based
selection, and the BN statistics are then recomputed on that stage's training data.
"""

import math

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from .constants import CHANNELS, TRIAL_SAMPLES
from .data import load_training
from .model import CrossRhythm
from .preprocessing import apply_preprocessing, fit_preprocessing
from .utils import seed_everything

DEFAULTS = {
    "width": 80,
    "pretraining_epochs": 100,
    "calibration_epochs": 200,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "warmup_epochs": 5,
    "minimum_learning_rate_ratio": 0.01,
    "label_smoothing": 0.1,
    "ema_decay": 0.99,
    "max_time_shift": 0,
}


def learning_rate(base, epoch, epochs, warmup, minimum_ratio):
    """Linear warm-up for `warmup` epochs, then cosine decay to base * minimum_ratio."""
    if epoch <= warmup:
        return base * epoch / warmup
    progress = (epoch - warmup) / (epochs - warmup)
    return base * (minimum_ratio + (1 - minimum_ratio) * (1 + math.cos(math.pi * progress)) / 2)


def batches(n, batch_size, device, shuffle):
    """Index batches; a final batch of size one is merged into the previous batch."""
    order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    parts = list(order.split(batch_size))
    if len(parts) > 1 and len(parts[-1]) == 1:
        tail = parts.pop()
        parts[-1] = torch.cat((parts[-1], tail))
    return parts


def random_time_shift(x, max_shift):
    """Shift each trial by a random number of samples with reflection at the borders.

    Offsets are drawn even when `max_shift` is zero, so the random-number sequence
    of a run does not depend on this setting.
    """
    offsets = torch.randint(-max_shift, max_shift + 1, (len(x),), device=x.device)
    if max_shift == 0:
        return x
    samples = x.shape[-1]
    positions = torch.arange(samples, device=x.device)[None, :] - offsets[:, None]
    reflected = positions.remainder(2 * (samples - 1))
    indices = torch.where(reflected < samples, reflected, 2 * (samples - 1) - reflected)
    return x.gather(2, indices[:, None, :].expand(-1, x.shape[1], -1))


@torch.no_grad()
def recompute_batchnorm(model, x, batch_size):
    """Recompute the BN statistics of `model` on training data (dropout disabled)."""
    model.eval()
    layers = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [layer.momentum for layer in layers]
    for layer in layers:
        layer.reset_running_stats()
        layer.train()
    seen = 0
    for indices in batches(len(x), batch_size, x.device, shuffle=False):
        seen += len(indices)
        for layer in layers:
            layer.momentum = len(indices) / seen  # cumulative average over batches
        model(x[indices])
    for layer, momentum in zip(layers, momenta):
        layer.momentum = momentum
    model.eval()


def train_stage(model, x, y, epochs, settings, log=print, stage=""):
    """Train one stage and load the EMA weights into `model`."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    average = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(settings["ema_decay"]))
    for epoch in range(1, epochs + 1):
        rate = learning_rate(settings["learning_rate"], epoch, epochs, settings["warmup_epochs"],
                             settings["minimum_learning_rate_ratio"])
        for group in optimizer.param_groups:
            group["lr"] = rate
        model.train()
        total = 0.0
        for indices in batches(len(y), settings["batch_size"], y.device, shuffle=True):
            inputs = random_time_shift(x[indices], settings["max_time_shift"])
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.training_loss(inputs, y[indices], settings["label_smoothing"])
            loss.backward()
            optimizer.step()
            model.project_weights()
            average.update_parameters(model)
            total += float(loss.detach()) * len(indices)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            log(f"{stage} epoch {epoch:3d}/{epochs}  loss {total / len(y):.4f}  lr {rate:.2e}")
    recompute_batchnorm(average.module, x, settings["batch_size"])
    model.load_state_dict(average.module.state_dict(), strict=True)


def train_crossrhythm(dataset, subject, seed, device="cuda", rhythm_pathway=True, settings=None, log=print):
    """Train CrossRhythm for one target participant and return (model, preprocessing, config)."""
    settings = {**DEFAULTS, **(settings or {})}
    seed_everything(seed)
    device = torch.device(device)
    data = load_training(dataset, subject)
    config = {"name": "CrossRhythm", "channels": len(CHANNELS[dataset]), "samples": TRIAL_SAMPLES,
              "n_classes": data["n_classes"], "width": settings["width"], "rhythm_pathway": rhythm_pathway}
    model = CrossRhythm(**{key: value for key, value in config.items() if key != "name"}).to(device)
    preprocessing = fit_preprocessing(data, "CrossRhythm")

    for stage, role, epochs in (("pretraining", "source_train", settings["pretraining_epochs"]),
                                ("calibration", "calibration", settings["calibration_epochs"])):
        group = data[role]
        x = apply_preprocessing(group["X"], group["trials"].subject, preprocessing)
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        y = torch.as_tensor(group["y"], dtype=torch.long, device=device)
        log(f"{stage}: {len(y)} trials")
        train_stage(model, x, y, epochs, settings, log, stage)
    model.eval()
    return model, preprocessing, config
