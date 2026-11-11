"""Recompute the result tables of the paper from the stored per-trial predictions.

    python scripts/reproduce_tables.py

Input: results/predictions.csv.gz (every test trial of every trained model).
Output (results/tables/):
    table2_accuracy.csv      accuracy (mean and SD across participants) per dataset
    table3_statistics.csv    paired comparison of CrossRhythm with each competing decoder
    table4_participants.csv  accuracy per participant (mean and SD across seeds), causal setting
    table5_ablation.csv      gain provided by the rhythm pathway (first five seeds)

Aggregation: metrics are computed per test session, averaged over the sessions
of a participant, then over seeds, and finally over participants.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossrhythm.constants import DATASETS, N_CLASSES, RESULTS_DIR, SUBJECTS  # noqa: E402
from crossrhythm.utils import classification_metrics  # noqa: E402

SEEDS = (17, 42, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033)
ABLATION_SEEDS = SEEDS[:5]
COMPETING = ("EEGNet", "GAT", "EEG-SimpleConv", "MSVTNet", "DMSACNN")
BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED = 20000, 20261031
TOLERANCE = 1e-10  # differences smaller than this (in percentage points) count as ties
PARTICIPANTS = [(dataset, subject) for dataset in DATASETS for subject in SUBJECTS[dataset]]


def per_fit_metrics(predictions):
    """Metrics per trained model (policy, model, dataset, subject, seed), sessions weighted equally."""
    rows = []
    keys = ["policy", "model", "dataset", "subject", "seed", "session"]
    for key, group in predictions.groupby(keys, sort=False):
        rows.append(dict(zip(keys, key), **classification_metrics(group.y_true, group.y_pred,
                                                                   N_CLASSES[key[2]])))
    sessions = pd.DataFrame(rows)
    return sessions.groupby(keys[:-1], as_index=False)[["accuracy", "balanced_accuracy", "macro_f1", "kappa"]].mean()


def per_participant(fits, model, policy, seeds):
    """Accuracy (in %) of each participant, averaged over the given seeds, in a fixed order."""
    local = fits.loc[fits.model.eq(model) & fits.policy.eq(policy) & fits.seed.isin(seeds)]
    means = local.groupby(["dataset", "subject"])[["accuracy", "kappa"]].mean()
    return means.loc[PARTICIPANTS]


def table_accuracy(fits):
    rows = []
    for policy in ("CAUSAL", "STATIC"):
        for model in (*COMPETING, "CrossRhythm"):
            values = per_participant(fits, model, policy, SEEDS)
            row = {"setting": policy.lower(), "decoder": model}
            for dataset in DATASETS:
                accuracy = 100 * values.loc[dataset].accuracy
                row[f"{dataset}_mean"], row[f"{dataset}_sd"] = accuracy.mean(), accuracy.std(ddof=1)
            row["pooled_mean"], row["pooled_sd"] = 100 * values.accuracy.mean(), 100 * values.accuracy.std(ddof=1)
            row["pooled_kappa"] = values.kappa.mean()
            rows.append(row)
    return pd.DataFrame(rows)


def wilcoxon_test(gain):
    """Two-sided Wilcoxon signed-rank test (normal approximation, no continuity correction)."""
    values = np.where(np.abs(gain) <= TOLERANCE, 0, np.round(gain, 10))
    if not np.count_nonzero(values):
        return 1.0
    return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided",
                          method="approx", correction=False).pvalue)


def holm(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values, kind="stable")
    adjusted = np.maximum.accumulate(p_values[order] * np.arange(len(p_values), 0, -1))
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1)
    return result


def bootstrap_indices():
    """Participants are resampled with replacement within each dataset."""
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    blocks, offset = [], 0
    for dataset in DATASETS:
        size = len(SUBJECTS[dataset])
        blocks.append(rng.integers(offset, offset + size, size=(BOOTSTRAP_REPLICATES, size)))
        offset += size
    return np.concatenate(blocks, axis=1)


def table_statistics(fits):
    indices = bootstrap_indices()
    rows = []
    for policy in ("CAUSAL", "STATIC"):
        proposed = 100 * per_participant(fits, "CrossRhythm", policy, SEEDS).accuracy.to_numpy()
        family = []
        for model in COMPETING:
            gain = proposed - 100 * per_participant(fits, model, policy, SEEDS).accuracy.to_numpy()
            low, high = np.percentile(gain[indices].mean(axis=1), [2.5, 97.5])
            family.append({"setting": policy.lower(), "decoder": model, "gain_pp": gain.mean(),
                           "ci95_low_pp": low, "ci95_high_pp": high,
                           "wins": int((gain > TOLERANCE).sum()), "p_value": wilcoxon_test(gain)})
        for row, adjusted in zip(family, holm([row["p_value"] for row in family])):
            row["p_holm"] = adjusted
        rows.extend(family)
    return pd.DataFrame(rows)


def table_participants(fits):
    local = fits.loc[fits.policy.eq("CAUSAL") & fits.seed.isin(SEEDS)]
    summary = local.groupby(["model", "dataset", "subject"]).accuracy.agg(["mean", "std"]) * 100
    table = summary.unstack("model")
    table.columns = [f"{model}_{statistic}" for statistic, model in table.columns]
    order = [f"{model}_{statistic}" for model in (*COMPETING, "CrossRhythm") for statistic in ("mean", "std")]
    return table.loc[PARTICIPANTS, order].reset_index()


def table_ablation(fits):
    rows = []
    for policy in ("CAUSAL", "STATIC"):
        full = 100 * per_participant(fits, "CrossRhythm", policy, ABLATION_SEEDS).accuracy
        reduced = 100 * per_participant(fits, "CrossRhythm-NoRhythm", policy, ABLATION_SEEDS).accuracy
        gain = (full - reduced)
        for dataset in (*DATASETS, "pooled"):
            values = gain.to_numpy() if dataset == "pooled" else gain.loc[dataset].to_numpy()
            rows.append({"setting": policy.lower(), "dataset": dataset, "gain_pp": values.mean(),
                         "improved_participants": int((values > TOLERANCE).sum()),
                         "participants": len(values), "p_value": wilcoxon_test(values)})
    return pd.DataFrame(rows)


def main():
    predictions = pd.read_csv(RESULTS_DIR / "predictions.csv.gz",
                              usecols=["policy", "dataset", "model", "subject", "seed", "session", "y_true", "y_pred"])
    fits = per_fit_metrics(predictions)
    output = RESULTS_DIR / "tables"
    output.mkdir(parents=True, exist_ok=True)
    tables = {"table2_accuracy": table_accuracy(fits), "table3_statistics": table_statistics(fits),
              "table4_participants": table_participants(fits), "table5_ablation": table_ablation(fits)}
    pd.set_option("display.width", 200)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.6g")
        print(f"\n{name}\n{table.round(3).to_string(index=False)}")
    print(f"\nTables written to {output}")


if __name__ == "__main__":
    main()
