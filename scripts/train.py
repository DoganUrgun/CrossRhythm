"""Train CrossRhythm for one or all participants of a dataset.

Examples:
    python scripts/train.py --dataset zhou2016 --subject 1 --seed 17
    python scripts/train.py --dataset bciciv2a --subject all --seed 2029
    python scripts/train.py --dataset bciciv2b --subject 3 --seed 17 --no-rhythm-pathway

The paper uses the seeds 17, 42 and 2026-2033. One model takes roughly 40 min
on a single GPU. Checkpoints are written to runs/ and can be evaluated with
scripts/evaluate.py --checkpoint <file>.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossrhythm.checkpoints import save_checkpoint  # noqa: E402
from crossrhythm.constants import DATASETS, ROOT, SUBJECTS  # noqa: E402
from crossrhythm.training import train_crossrhythm  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--subject", default="all", help="participant number or 'all'")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--no-rhythm-pathway", action="store_true",
                        help="train the variant without the rhythm pathway (ablation)")
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    args = parser.parse_args()

    subjects = SUBJECTS[args.dataset] if args.subject == "all" else [int(args.subject)]
    suffix = "_no_rhythm" if args.no_rhythm_pathway else ""
    for subject in subjects:
        print(f"Training {args.dataset} participant {subject}, seed {args.seed}", flush=True)
        model, preprocessing, config = train_crossrhythm(
            args.dataset, subject, args.seed, args.device, rhythm_pathway=not args.no_rhythm_pathway,
            log=lambda message: print(message, flush=True))
        path = args.output / f"{args.dataset}_s{subject:02d}_seed{args.seed}{suffix}.pt"
        save_checkpoint(path, model, config, preprocessing, args.dataset, subject, args.seed)
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
