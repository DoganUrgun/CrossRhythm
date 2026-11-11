"""Load the fixed data splits and the prepared EEG arrays.

For each dataset, `splits/<dataset>/` contains two tables:

* earlier_sessions.csv  trials of the earlier sessions with their role
  (`train` or `validation`; 80/20 split, stratified by session and class) and an
  additional inner split used only by the MSVTNet and DMSACNN training recipes.
* later_sessions.csv    trials of the later sessions (test data).

For a target participant, the earlier `train` trials of all other participants
form the source set, the earlier `train` trials of the target form the
calibration set, and the later sessions of the target form the test set.
"""

import numpy as np
import pandas as pd

from .constants import CHANNELS, N_CLASSES, ROOT, SPLITS_DIR, SUBJECTS


def chronological(rows):
    """Sort trials by participant, session, run, and onset (recording order)."""
    rows = rows.copy()
    runs = rows["run"].astype(str)
    rows["_run"] = pd.to_numeric(runs, errors="coerce").fillna(runs.map({"A": 0, "B": 1}))
    rows = rows.sort_values(["subject", "session", "_run", "raw_onset_sample", "array_index", "trial_id"],
                            kind="stable")
    return rows.drop(columns="_run").reset_index(drop=True)


def read_split(dataset, part):
    """Read `earlier_sessions` or `later_sessions` of a dataset."""
    path = SPLITS_DIR / dataset / f"{part}.csv"
    rows = pd.read_csv(path, dtype={"trial_id": str, "run": str, "prepared_file": str}, keep_default_na=False)
    return chronological(rows)


def load_arrays(rows, dataset):
    """Load the EEG of the given trials from data/prepared (float32, [trials, channels, samples])."""
    output = np.empty((len(rows), len(CHANNELS[dataset]), 1000), dtype=np.float32)
    for pointer, group in rows.groupby("prepared_file", sort=False):
        path = ROOT / pointer
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing. Run scripts/prepare_data.py first.")
        array = np.load(path, mmap_mode="r")
        output[group.index.to_numpy()] = array[group.array_index.to_numpy(dtype=np.int64)]
    return output


def _group(rows, dataset, with_labels=True):
    rows = rows.reset_index(drop=True)
    group = {"X": load_arrays(rows, dataset), "trials": rows.drop(columns=["y"], errors="ignore")}
    if with_labels:
        group["y"] = rows["y"].to_numpy(dtype=np.int64).copy()
    return group


def load_training(dataset, target):
    """Source and calibration data (with their validation parts) for one target participant."""
    if int(target) not in SUBJECTS[dataset]:
        raise ValueError(f"Unknown participant {target} for {dataset}")
    rows = read_split(dataset, "earlier_sessions")
    is_target = rows.subject.eq(int(target))
    roles = {"source_train": ~is_target & rows.role.eq("train"),
             "source_validation": ~is_target & rows.role.eq("validation"),
             "calibration": is_target & rows.role.eq("train"),
             "target_validation": is_target & rows.role.eq("validation")}
    data = {"dataset": dataset, "target": int(target), "n_classes": N_CLASSES[dataset]}
    for name, mask in roles.items():
        data[name] = _group(rows.loc[mask], dataset)
    return data


def load_calibration_inputs(dataset, target):
    """EEG of the target's calibration trials, without labels (needed by the causal setting)."""
    rows = read_split(dataset, "earlier_sessions")
    rows = rows.loc[rows.subject.eq(int(target)) & rows.role.eq("train")]
    return _group(rows, dataset, with_labels=False)


def load_test(dataset, target):
    """EEG and labels of the target's later sessions."""
    rows = read_split(dataset, "later_sessions")
    return _group(rows.loc[rows.subject.eq(int(target))], dataset)
