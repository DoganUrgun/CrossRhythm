"""Input preprocessing fitted on training data only.

Every decoder receives the same 4-s trials and then applies its own
preprocessing. CrossRhythm (and EEG-SimpleConv) use:

* Zhou2016 only: a causal third-order Butterworth high-pass filter at 0.5 Hz.
* Euclidean alignment (EA): each participant's trials are multiplied by the
  inverse square root of that participant's mean (uncentered) covariance,
  estimated on the participant's training trials.
* Channel-wise standardization, fitted once on the aligned source trials and
  once on the aligned calibration trials of the target participant.

The fitted state is a plain dictionary that is stored in each checkpoint.
"""

import numpy as np
from scipy.signal import butter, cheby1, filtfilt, lfilter, sosfiltfilt


def filter_settings(decoder, dataset):
    """Fixed band filter used by each decoder (as in the original publications)."""
    if decoder == "EEGNet":
        return dict(family="butter", order=3, cutoff_hz=[4, 38], kind="bandpass", phase="causal")
    if decoder == "GAT":
        return dict(family="cheby1", order=6, ripple_db=1, cutoff_hz=[4, 40], kind="bandpass", phase="zero")
    if decoder in ("CrossRhythm", "EEG-SimpleConv") and dataset == "zhou2016":
        return dict(family="butter", order=3, cutoff_hz=[0.5], kind="highpass", phase="causal")
    return dict(family="none")


def filter_epochs(x, settings, sample_rate=250):
    """Filter each trial independently along time (float64 output)."""
    x = np.array(x, dtype=np.float64, copy=True)
    if settings["family"] == "none":
        return x
    cutoff = np.asarray(settings["cutoff_hz"], dtype=float) / (sample_rate / 2)
    cutoff = cutoff.item() if cutoff.size == 1 else cutoff
    if settings["family"] == "cheby1":
        sos = cheby1(settings["order"], settings["ripple_db"], cutoff, btype=settings["kind"], output="sos")
        return sosfiltfilt(sos, x, axis=-1)
    b, a = butter(settings["order"], cutoff, btype=settings["kind"])
    return lfilter(b, a, x, axis=-1) if settings["phase"] == "causal" else filtfilt(b, a, x, axis=-1)


def inverse_sqrt(covariance):
    """Symmetric inverse square root with a small eigenvalue floor."""
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(covariance)
    floor = max(float(values.max()) * 1e-10, 1e-12)
    return (vectors * np.maximum(values, floor) ** -0.5) @ vectors.T


def mean_covariance(x):
    """Mean of the uncentered trial covariances X X^T."""
    return sum(trial @ trial.T for trial in x) / len(x)


def channel_moments(x, ddof=0, scalar=False):
    """Mean and standard deviation per channel (or one scalar pair)."""
    axes = None if scalar else (0, 2)
    mean, std = np.mean(x, axis=axes), np.std(x, axis=axes, ddof=ddof)
    if not scalar:
        mean, std = mean[:, None], std[:, None]
    return dict(mean=np.asarray(mean, dtype=np.float32),
                std=np.asarray(np.maximum(std, 1e-6), dtype=np.float32))


def fit_preprocessing(data, decoder):
    """Fit the preprocessing of `decoder` on the source and calibration training sets.

    `data` is the dictionary returned by `crossrhythm.data.load_training`.
    """
    dataset, target = data["dataset"], int(data["target"])
    source, calibration = data["source_train"], data["calibration"]
    state = dict(decoder=decoder, dataset=dataset, target_subject=target,
                 filter=filter_settings(decoder, dataset), alignment={}, normalization={},
                 normalization_scope="none",
                 calibration_trial_ids=calibration["trials"].trial_id.astype(str).tolist())
    if decoder == "DMSACNN":
        return state  # DMSACNN receives the signals in microvolts without further processing.

    source_x = filter_epochs(source["X"], state["filter"])
    target_x = filter_epochs(calibration["X"], state["filter"])
    source_subjects = source["trials"].subject.to_numpy()

    if decoder in ("CrossRhythm", "EEG-SimpleConv"):
        aligned_source = np.empty_like(source_x)
        for subject in np.unique(source_subjects):
            mask = source_subjects == subject
            matrix = inverse_sqrt(mean_covariance(source_x[mask]))
            state["alignment"][str(int(subject))] = matrix
            aligned_source[mask] = np.einsum("ij,njt->nit", matrix, source_x[mask])
        state["alignment"][str(target)] = inverse_sqrt(mean_covariance(target_x))
        aligned_target = np.einsum("ij,njt->nit", state["alignment"][str(target)], target_x)
        state["normalization_scope"] = "source_and_target"
        state["normalization"]["source"] = channel_moments(aligned_source, ddof=1)
        state["normalization"]["target"] = channel_moments(aligned_target, ddof=1)
        return state

    # EEGNet and GAT: per-participant channel standardization; MSVTNet: one scalar pair.
    scalar = decoder == "MSVTNet"
    state["normalization_scope"] = "per_subject"
    for subject in np.unique(source_subjects):
        state["normalization"][str(int(subject))] = channel_moments(source_x[source_subjects == subject], scalar=scalar)
    state["normalization"][str(target)] = channel_moments(target_x, scalar=scalar)
    return state


def apply_preprocessing(x, subjects, state, alignment=None):
    """Apply a fitted state to trials of the given participants (float32 output).

    `alignment` optionally replaces the stored EA matrix of the target participant
    (used by the causal setting, in which the EA reference is updated online).
    """
    x = filter_epochs(x, state["filter"])
    subjects = np.asarray(subjects).astype(int)
    for subject in np.unique(subjects):
        mask = subjects == subject
        key = str(int(subject))
        matrix = alignment if (alignment is not None and int(subject) == state["target_subject"]) \
            else state["alignment"].get(key)
        if matrix is not None:
            x[mask] = np.einsum("ij,njt->nit", matrix, x[mask])
        if state["normalization_scope"] == "source_and_target":
            key = "target" if int(subject) == state["target_subject"] else "source"
        if state["normalization_scope"] != "none":
            moments = state["normalization"][key]
            x[mask] = (x[mask] - moments["mean"]) / moments["std"]
    return x.astype(np.float32)
