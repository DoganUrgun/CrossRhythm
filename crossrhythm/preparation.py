"""Download the original recordings and extract the 4-s motor imagery trials.

Sources (see configs/data_sources.json for URLs and SHA-256 checksums):
* BCI Competition IV 2a and 2b (Graz University of Technology), MATLAB files.
* Zhou2016 (Zhou et al., PLoS ONE, 2016), Neuroscan CNT files in one archive.

For every recording, `data/prepared/<dataset>/<recording>/` receives
* X.npy        float32 trials [trials, channels, 1000] in microvolts,
* trials.csv   one row per cue with label, onset and the row index in X.npy.

Trials start at the motor imagery cue and last 4 s (1000 samples at 250 Hz).
No filtering or normalization is applied here; each decoder preprocesses its
input itself. Every written X.npy is checked against configs/prepared_checksums.csv.
"""

import hashlib
import json
import re
import time
import zipfile

import numpy as np
import pandas as pd

from .constants import CHANNELS, CLASSES, CONFIG_DIR, DATA_DIR

CHUNK = 8 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def data_sources():
    return json.loads((CONFIG_DIR / "data_sources.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------- download

def download(url, path, expected_bytes, expected_sha256):
    """Download a file in chunks (resumable) and verify its checksum."""
    import requests

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    with requests.Session() as session, partial.open("ab") as output:
        while offset < expected_bytes:
            stop = min(offset + CHUNK, expected_bytes) - 1
            for attempt in range(4):
                try:
                    response = session.get(url, headers={"Range": f"bytes={offset}-{stop}"}, timeout=(30, 120))
                    response.raise_for_status()
                    block = response.content
                    if len(block) != stop - offset + 1:
                        raise ValueError("incomplete block")
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            output.write(block)
            offset = stop + 1
    if sha256(partial) != expected_sha256:
        partial.unlink()
        raise ValueError(f"Checksum mismatch for {path.name}; please run the download again.")
    partial.replace(path)


def download_dataset(dataset):
    sources = data_sources()
    folder = DATA_DIR / "raw" / dataset
    if dataset == "zhou2016":
        archive = next(row for row in sources["archives"] if row["dataset"] == "zhou2016")
        path = folder / archive["name"]
        if not path.exists():
            print(f"Downloading {archive['url']} ...", flush=True)
            download(archive["url"], path, archive["bytes"], archive["sha256"])
        with zipfile.ZipFile(path) as source:
            for record in (row for row in sources["recordings"] if row["dataset"] == "zhou2016"):
                destination = folder / record["name"]
                if not destination.exists():
                    with source.open("data/" + record["name"]) as stream, destination.open("wb") as output:
                        for block in iter(lambda: stream.read(CHUNK), b""):
                            output.write(block)
                if sha256(destination) != record["sha256"]:
                    raise ValueError(f"Checksum mismatch for {destination}")
        return
    for record in (row for row in sources["recordings"] if row["dataset"] == dataset):
        path = folder / record["name"]
        if not path.exists():
            print(f"Downloading {record['url']} ...", flush=True)
            download(record["url"], path, record["bytes"], record["sha256"])
        elif sha256(path) != record["sha256"]:
            raise ValueError(f"Checksum mismatch for {path}")


# ---------------------------------------------------------------------- readers

def _trial_row(dataset, uid, subject, session, run, trial, label, artifact, cacheable, reason, index, onset, start, stop):
    return dict(trial_id=uid, dataset=dataset, subject=subject, session=session, run=run, trial=trial,
                label_raw=int(label), y=int(label) - 1, artifact_flag=bool(artifact), cacheable=cacheable,
                retained=reason == "retained", reason=reason, array_index=index if cacheable else -1,
                raw_onset_sample=int(onset), window_start_sample=start, window_stop_sample=stop)


def _reason(valid, complete, finite, artifact):
    if not valid:
        return "invalid_label"
    if not complete:
        return "insufficient_window"
    if not finite:
        return "nonfinite_eeg"
    return "supplied_artifact_flag" if artifact else "retained"


def read_bciciv2a(path):
    """One BCI IV 2a file (T: training session 1, E: evaluation session 2)."""
    from scipy.io import loadmat

    subject, session = int(path.name[1:3]), 1 if path.stem[-1] == "T" else 2
    raw = loadmat(path, simplify_cells=True, variable_names=["data"])
    rows, trials, index, imagery_run = [], [], 0, 0
    for run, record in enumerate(np.atleast_1d(raw["data"])):
        x = np.asarray(record["X"])
        onsets, labels = np.atleast_1d(record["trial"]), np.atleast_1d(record["y"])
        artifacts = np.atleast_1d(record["artifacts"])
        if len(onsets):
            imagery_run += 1
        for trial, (onset, label, artifact) in enumerate(zip(onsets, labels, artifacts), 1):
            # The MATLAB onset marks the fixation cross; the imagery cue follows 2 s later.
            start, stop = int(onset) - 1 + 500, int(onset) - 1 + 1500
            complete = start >= 0 and stop <= len(x)
            finite = complete and np.isfinite(x[start:stop, :22]).all()
            reason = _reason(label in (1, 2, 3, 4), complete, finite, artifact)
            cacheable = reason in ("retained", "supplied_artifact_flag")
            uid = f"bciciv2a:S{subject:02d}:ses{session:02d}:run{run:02d}:trial{trial:03d}"
            row = _trial_row("bciciv2a", uid, subject, session, run, trial, label, artifact,
                             cacheable, reason, index, onset, start, stop)
            row["imagery_run"] = imagery_run
            rows.append(row)
            if cacheable:
                trials.append(np.asarray(x[start:stop, :22], dtype=np.float64).T.astype(np.float32))
                index += 1
    return [(path.stem, pd.DataFrame(rows), trials)]


def read_bciciv2b(path):
    """One BCI IV 2b file; T contains sessions 1-3 and E contains sessions 4-5."""
    from scipy.io import loadmat

    subject = int(path.name[1:3])
    sessions = [1, 2, 3] if path.stem[-1] == "T" else [4, 5]
    raw = loadmat(path, simplify_cells=True, variable_names=["data"])
    output = []
    for session, record in zip(sessions, np.atleast_1d(raw["data"])):
        x = np.asarray(record["X"])
        labels, onsets = np.atleast_1d(record["y"]), np.atleast_1d(record["trial"])
        artifacts = np.atleast_1d(record["artifacts"])
        rows, trials, index = [], [], 0
        for trial, (onset, label, artifact) in enumerate(zip(onsets, labels, artifacts), 1):
            # The MATLAB onset marks the start of the trial; the imagery cue follows 3 s later.
            start, stop = int(onset) - 1 + 750, int(onset) - 1 + 1750
            complete = start >= 0 and stop <= len(x)
            finite = complete and np.isfinite(x[start:stop, :3]).all()
            reason = _reason(label in (1, 2), complete, finite, artifact)
            cacheable = reason in ("retained", "supplied_artifact_flag")
            uid = f"bciciv2b:S{subject:02d}:ses{session:02d}:trial{trial:03d}"
            rows.append(_trial_row("bciciv2b", uid, subject, session, 0, trial, label, artifact,
                                   cacheable, reason, index, onset, start, stop))
            if cacheable:
                trials.append(np.asarray(x[start:stop, :3], dtype=np.float64).T.astype(np.float32))
                index += 1
        output.append((f"{path.stem}_session{session:02d}", pd.DataFrame(rows), trials))
    return output


def read_zhou2016(path):
    """One Zhou2016 run (Neuroscan CNT, 16-bit); the two EOG channels are excluded."""
    import mne

    match = re.fullmatch(r"S([1-4])_([1-3])([AB])\.cnt", path.name)
    subject, session, run = int(match[1]), int(match[2]), match[3]
    raw = mne.io.read_raw_cnt(str(path), preload=True, eog=["VEOU", "VEOL"], data_format="int16", verbose="ERROR")
    events, _ = mne.events_from_annotations(raw, event_id={"1": 1, "2": 2, "3": 3}, regexp=None,
                                            use_rounding=True, verbose=False)
    x = raw.get_data(picks=list(range(2, 16))).T * 1e6  # volts -> microvolts
    rows, trials, index = [], [], 0
    for trial, event in enumerate(events, 1):
        start, label = int(event[0] - raw.first_samp), int(event[2])
        stop = start + 1000
        complete = start >= 0 and stop <= len(x)
        finite = complete and np.isfinite(x[start:stop]).all()
        reason = _reason(label in (1, 2, 3), complete, finite, False)
        cacheable = reason == "retained"
        uid = f"zhou2016:S{subject:02d}:ses{session:02d}:run{run}:trial{trial:03d}"
        rows.append(_trial_row("zhou2016", uid, subject, session, run, trial, label, False,
                               cacheable, reason, index, start, start, stop))
        if cacheable:
            trials.append(np.asarray(x[start:stop], dtype=np.float64).T.astype(np.float32))
            index += 1
    raw.close()
    return [(path.stem, pd.DataFrame(rows), trials)]


READERS = {"bciciv2a": read_bciciv2a, "bciciv2b": read_bciciv2b, "zhou2016": read_zhou2016}


def raw_files(dataset):
    sources = data_sources()["recordings"]
    return [DATA_DIR / "raw" / dataset / row["name"] for row in sources if row["dataset"] == dataset]


# ------------------------------------------------------------------- preparation

def prepare_dataset(dataset):
    """Write X.npy, trials.csv and metadata.json for every recording of a dataset."""
    checksums = pd.read_csv(CONFIG_DIR / "prepared_checksums.csv")
    expected = dict(zip(checksums.recording[checksums.dataset == dataset],
                        checksums.x_sha256[checksums.dataset == dataset]))
    for path in raw_files(dataset):
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run with --download first.")
        for recording, rows, trials in READERS[dataset](path):
            folder = DATA_DIR / "prepared" / dataset / recording
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / "X.npy", np.stack(trials).astype(np.float32), allow_pickle=False)
            rows["prepared_file"] = f"data/prepared/{dataset}/{recording}/X.npy"
            rows.to_csv(folder / "trials.csv", index=False)
            (folder / "metadata.json").write_text(json.dumps({
                "dataset": dataset, "recording": recording, "channels": CHANNELS[dataset],
                "classes": CLASSES[dataset], "sample_rate_hz": 250, "units": "microvolt",
                "window": "0-4 s after the motor imagery cue"}, indent=2) + "\n", encoding="utf-8")
            if sha256(folder / "X.npy") != expected[recording]:
                raise ValueError(f"{dataset}/{recording}: the prepared EEG differs from the reference checksum. "
                                 "Please check the package versions in requirements.txt.")
            print(f"Prepared {dataset}/{recording} ({len(trials)} trials)", flush=True)


def verify_dataset(dataset):
    """Check every prepared X.npy of a dataset against the reference checksums."""
    checksums = pd.read_csv(CONFIG_DIR / "prepared_checksums.csv")
    rows = checksums[checksums.dataset == dataset]
    for row in rows.itertuples():
        path = DATA_DIR / "prepared" / dataset / row.recording / "X.npy"
        if not path.exists() or sha256(path) != row.x_sha256:
            raise ValueError(f"{path} is missing or differs from the reference checksum.")
    print(f"Verified {dataset}: {len(rows)} recordings", flush=True)
