"""Evaluate trained CrossRhythm models on the later sessions.

Examples:
    # One released model, both settings
    python scripts/evaluate.py --dataset zhou2016 --subject 1

    # All released models of a dataset
    python scripts/evaluate.py --dataset bciciv2a --subject all

    # A model trained with scripts/train.py
    python scripts/evaluate.py --dataset zhou2016 --subject 1 --checkpoint runs/zhou2016_s01_seed17.pt

The script writes per-trial predictions and prints the accuracy per participant.
For the released models it also reports whether the predictions agree with the
stored predictions in results/predictions.csv.gz.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossrhythm.checkpoints import load_checkpoint  # noqa: E402
from crossrhythm.constants import CHECKPOINT_DIR, DATASETS, N_CLASSES, RESULTS_DIR, ROOT, SUBJECTS  # noqa: E402
from crossrhythm.data import load_calibration_inputs, load_test  # noqa: E402
from crossrhythm.inference import predict_causal, predict_static  # noqa: E402
from crossrhythm.utils import classification_metrics  # noqa: E402


def evaluate_subject(dataset, subject, checkpoint_path, settings, device):
    model, preprocessing, checkpoint = load_checkpoint(checkpoint_path, device)
    test = load_test(dataset, subject)
    calibration = load_calibration_inputs(dataset, subject)
    rows = []
    for setting in settings:
        if setting == "static":
            probabilities = predict_static(model, test, preprocessing, device)
        else:
            probabilities = predict_causal(model, test, calibration, preprocessing, device)
        frame = test["trials"][["trial_id", "subject", "session"]].copy()
        frame.insert(0, "setting", setting)
        frame.insert(1, "dataset", dataset)
        frame["seed"] = checkpoint["seed"]
        frame["y_true"] = test["y"]
        frame["y_pred"] = probabilities.argmax(1)
        for index in range(probabilities.shape[1]):
            frame[f"p_{index}"] = probabilities[:, index]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def summarize(predictions, dataset):
    """Accuracy per participant: sessions are averaged with equal weight."""
    summary = []
    for (setting, subject), group in predictions.groupby(["setting", "subject"]):
        sessions = [classification_metrics(part.y_true, part.y_pred, N_CLASSES[dataset])["accuracy"]
                    for _, part in group.groupby("session")]
        summary.append({"setting": setting, "subject": subject, "accuracy_%": 100 * np.mean(sessions)})
    return pd.DataFrame(summary)


def compare_with_stored(predictions):
    """Agreement of the predicted classes with results/predictions.csv.gz (released models only)."""
    stored = pd.read_csv(RESULTS_DIR / "predictions.csv.gz", dtype={"trial_id": str},
                         usecols=["policy", "model", "seed", "trial_id", "y_pred"])
    stored = stored.loc[stored.model.eq("CrossRhythm")]
    stored["setting"] = stored.policy.str.lower()
    merged = predictions.merge(stored, on=["setting", "seed", "trial_id"], suffixes=("", "_stored"))
    if len(merged) != len(predictions):
        return None
    return float((merged.y_pred == merged.y_pred_stored).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--subject", default="all", help="participant number or 'all'")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint file (default: released model)")
    parser.add_argument("--setting", choices=["static", "causal", "both"], default="both")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "evaluation")
    args = parser.parse_args()

    subjects = SUBJECTS[args.dataset] if args.subject == "all" else [int(args.subject)]
    if args.checkpoint and len(subjects) != 1:
        parser.error("--checkpoint requires a single --subject")
    settings = ["static", "causal"] if args.setting == "both" else [args.setting]

    frames = []
    for subject in subjects:
        path = args.checkpoint or CHECKPOINT_DIR / args.dataset / f"subject_{subject:02d}.pt"
        print(f"Evaluating {args.dataset} participant {subject} ({path.name}) ...", flush=True)
        frames.append(evaluate_subject(args.dataset, subject, path, settings, args.device))
    predictions = pd.concat(frames, ignore_index=True)

    args.output.mkdir(parents=True, exist_ok=True)
    name = f"{args.dataset}_{'all' if args.subject == 'all' else f'subject_{int(args.subject):02d}'}.csv"
    predictions.to_csv(args.output / name, index=False)
    print(summarize(predictions, args.dataset).round(2).to_string(index=False))
    print(f"Predictions written to {args.output / name}")
    if args.checkpoint is None:
        agreement = compare_with_stored(predictions)
        if agreement is not None:
            print(f"Agreement with stored predictions: {100 * agreement:.2f}% of trials")


if __name__ == "__main__":
    main()
