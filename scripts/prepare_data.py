"""Download the public datasets and prepare the trial arrays.

Examples:
    python scripts/prepare_data.py --download --prepare            # all datasets
    python scripts/prepare_data.py --dataset zhou2016 --download --prepare
    python scripts/prepare_data.py --verify                        # check existing arrays

The raw files are stored in data/raw/ (about 1.6 GB in total) and the prepared
arrays in data/prepared/ (about 0.6 GB).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossrhythm.constants import DATASETS  # noqa: E402
from crossrhythm.preparation import download_dataset, prepare_dataset, verify_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, help="default: all three datasets")
    parser.add_argument("--download", action="store_true", help="download the original recordings")
    parser.add_argument("--prepare", action="store_true", help="extract the trials into data/prepared/")
    parser.add_argument("--verify", action="store_true", help="verify the prepared arrays")
    args = parser.parse_args()
    if not (args.download or args.prepare or args.verify):
        parser.error("choose at least one of --download, --prepare, --verify")
    for dataset in [args.dataset] if args.dataset else DATASETS:
        if args.download:
            download_dataset(dataset)
        if args.prepare:
            prepare_dataset(dataset)
        if args.verify:
            verify_dataset(dataset)


if __name__ == "__main__":
    main()
