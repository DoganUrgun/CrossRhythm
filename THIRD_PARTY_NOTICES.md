# Third-party notices

This repository is released under the MIT License (see `LICENSE`). It builds on
the following published work.

## Code

| Component | Used in | Original source | License |
|---|---|---|---|
| EEGNet (Lawhern et al., 2018) | `crossrhythm/eegnet.py` (compact pathway) | [vlawhern/arl-eegmodels](https://github.com/vlawhern/arl-eegmodels) | U.S. Army Research Laboratory license (CC0 1.0 with patent grant, Apache 2.0); full text in `licenses/EEGNet-LICENSE.txt` |
| MSVTNet (Liu et al., 2024) | `crossrhythm/model.py` (multiscale stem, auxiliary heads, transformer layout) | [SheepTAO/dpeeg](https://github.com/SheepTAO/dpeeg) | MIT, Copyright (c) 2023 SheepTAO; full text in `licenses/MSVTNet-dpeeg-LICENSE.txt` |

The windowed-sinc filters of the rhythm pathway follow the formulation of
SincNet (Ravanelli and Bengio, 2018); no SincNet code is included.

## Datasets

The EEG recordings are not redistributed. `scripts/prepare_data.py` downloads
them from their original providers:

* BCI Competition IV, datasets 2a and 2b (Graz University of Technology),
  https://www.bbci.de/competition/iv/
* Zhou2016 (Zhou et al., PLoS ONE, 2016, https://doi.org/10.1371/journal.pone.0162657),
  distributed through figshare (download link in `configs/data_sources.json`)

Please cite the dataset publications listed in `README.md` when using the data.
