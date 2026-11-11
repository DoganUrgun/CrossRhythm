"""Dataset definitions and repository paths used throughout the package."""

from pathlib import Path

# Repository root (the folder that contains README.md).
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SPLITS_DIR = ROOT / "splits"
CONFIG_DIR = ROOT / "configs"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

DATASETS = ("bciciv2a", "bciciv2b", "zhou2016")

# Participants of each dataset.
SUBJECTS = {
    "bciciv2a": list(range(1, 10)),
    "bciciv2b": list(range(1, 10)),
    "zhou2016": list(range(1, 5)),
}

# Recording sessions used for training (earlier) and for testing (later).
EARLIER_SESSIONS = {"bciciv2a": [1], "bciciv2b": [1, 2, 3], "zhou2016": [1, 2]}
LATER_SESSIONS = {"bciciv2a": [2], "bciciv2b": [4, 5], "zhou2016": [3]}

CLASSES = {
    "bciciv2a": ["left_hand", "right_hand", "feet", "tongue"],
    "bciciv2b": ["left_hand", "right_hand"],
    "zhou2016": ["left_hand", "right_hand", "feet"],
}
N_CLASSES = {name: len(classes) for name, classes in CLASSES.items()}

CHANNELS = {
    "bciciv2a": ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
                 "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"],
    "bciciv2b": ["C3", "Cz", "C4"],
    "zhou2016": ["Fp1", "Fp2", "FC3", "FCz", "FC4", "C3", "Cz", "C4",
                 "CP3", "CPz", "CP4", "O1", "Oz", "O2"],
}

SAMPLE_RATE = 250   # Hz, all datasets
TRIAL_SAMPLES = 1000  # 4 s after the cue
