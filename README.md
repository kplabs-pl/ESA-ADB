<h1>ESA Anomaly Detection Benchmark</h1>

The European Space Agency Anomaly Detection Benchmark (ESA-ADB) consists of three main components (visualised in the figure below for easier comprehension):
1.	Large-scale, curated, structured, ML-ready ESA Anomalies Dataset (ESA-AD, in short) of real-life satellite telemetry collected from three ESA missions (out of which two are selected for benchmarking in ESA-ADB), manually annotated by spacecraft operations engineers (SOEs) and ML experts, and cross-verified using state-of-the-art algorithms. It can be downloaded from here: https://doi.org/10.5281/zenodo.12528696
2.	Evaluation pipeline designed by ML experts for the practical needs of SOEs from the ESA’s European Space Operations Centre (ESOC). It introduces new metrics designed for satellite telemetry according to the latest advancements in time series anomaly detection (TSAD) and simulates real operational scenarios, e.g. different mission phases and real-time monitoring. 
3.	Benchmarking results of TSAD algorithms selected and improved to comply with the space operations requirements.

<p align="center">
<img src="./ESA-ADB.svg"/>
</p>

We hope that this unique benchmark will allow researchers and scientists from academia, research institutes, national and international space agencies, and industry to validate models and approaches on a common baseline as well as research and develop novel, computational-efficient approaches for anomaly detection in satellite telemetry data.

The dataset results from the work of an 18-month project carried by an industry Consortium composed of Airbus Defence and Space, KP Labs, and the European Space Agency’s European Space Operations Centre. The project, funded by the European Space Agency (ESA) under the contract number 4000137682/22/D/SR, is a part of The Artificial Intelligence for Automation (A²I) Roadmap (De Canio et al., Development of an actionable AI roadmap for automating mission operations, 2023 SpaceOps Conference), a large endeavour started in 2021 to automate space operations by leveraging artificial intelligence.

The introduction below describes how to reproduce results presented in the ESA-ADB paper using the provided modified fork of the TimeEval framework.


## Initial requirements

- Some ESA-ADB functions work only on Linux (or Windows Subsystem for Linux 2), so we suggest using it as a development platform
- The hard disk must be configured to use NTFS file system
- It is recommended to have at least 512 GB free disk space to store artifacts from all experiments
- It is recommended to use Nvidia GPU with compute capability >= 7.1
- It is recommended to use a machine with at least 64 GB RAM (32 GB is an absolute minimum for the whole pipeline)

## Preparing datasets

Download raw ESA Anomalies Dataset from the link https://doi.org/10.5281/zenodo.12528696 and put ESA-Mission1 and ESA-Mission2 folders in the "data" folder.

### Generating preprocessed data for experiments

There are separate script to generate preprocessed data for TimeEval framework for each mission. The scripts are located in notebooks\data-prep folder.
- Mission1: Mission1_semisupervised_prep_from_raw.py
- Mission2: Mission2_semisupervised_prep_from_raw.py

The scripts generate all necessary files in data/preprocessed/multivariate folders and add records to data/preprocessed/datasets.csv if necessary (records for ESA-ADB are already added as a part of this repository)

## Environment setup

### Python environment
1. Install Anaconda environment manager, version >= 22.
2. Create a conda-environment and install all required dependencies.
   Use the file [`environment.yml`](./environment.yml) for this:
   `conda env create --file environment.yml`. Note that you should **not** install TimeEval from PyPI. Our repository contains the modified version of TimeEval in the "timeeval" folder.
3. Activate the new environment `conda activate timeeval`.

### Docker
1. Install Docker Engine, version >= 23.
2. Build Docker containers with algorithms of interest (e.g., listed in mission1_experiments.py) using instruction from README in the TimeEval-algorithms folder. 
   For our Telemanom-ESA, it is enough to run `sudo docker build -t registry.gitlab.hpi.de/akita/i/telemanom_esa ./telemanom_esa`.
   For our DC-VAE-ESA, it is enough to run `sudo docker build -t registry.gitlab.hpi.de/akita/i/dc_vae ./dc_vae`.

## Running benchmark
There is a separate script in the main folder of the repo to run a full grid of experiments for each mission:
- Mission1: mission1_experiments.py
- Mission2: mission2_experiments.py

The scripts configure and run all algorithms in Docker containers. Results are generated to 'results' folder

### Notes
- evaluation pipeline with novel time-aware metrics can only be run for datasets following the same structure as ESA Anomalies Dataset (with labels.csv and anomaly_types.csv)
- when analyzing results for different anomaly types for the lightweight subsets of channels, it is necessary to regenerate anomaly types using anomaly_types.csv using scripts/infer_anomaly_types.py. It is because anomaly types may depend on the analyzed subset of channels.
- for now, all algorithms treat rare nominal events as anomalies. To change that behaviour, it would be necessary to modify the code of the framework and some algorithms



<div align="center">
<img width="100px" src="https://github.com/TimeEval/TimeEval/raw/main/timeeval-icon.png" alt="TimeEval logo"/>
<h1 align="center">TimeEval</h1>
<p>
Evaluation Tool for Anomaly Detection Algorithms on Time Series.
</p>

[![CI](https://github.com/TimeEval/TimeEval/actions/workflows/build.yml/badge.svg)](https://github.com/TimeEval/TimeEval/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/timeeval/badge/?version=latest)](https://timeeval.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/TimeEval/TimeEval/branch/main/graph/badge.svg?token=esrQJQmMQe)](https://codecov.io/gh/TimeEval/TimeEval)
[![PyPI version](https://badge.fury.io/py/TimeEval.svg)](https://badge.fury.io/py/TimeEval)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![python version 3.7|3.8|3.9](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)

</div>

See [TimeEval Algorithms](https://gitlab.hpi.de/akita/timeeval-algorithms) (use [this link](https://github.com/TimeEval/TimeEval-algorithms) on Github) for algorithms that are compatible to this tool.
The algorithms in this repository are containerized and can be executed using the [`DockerAdapter`](./timeeval/adapters/docker.py) of TimeEval.

> If you use TimeEval, please consider [citing](#citation) our paper.

📖 TimeEval's documentation is hosted at https://timeeval.readthedocs.io.

## Features

- Large integrated benchmark dataset collection with more than 700 datasets
- Benchmark dataset interface to select datasets easily
- Adapter architecture for algorithm integration
  - JarAdapter
  - DistributedAdapter
  - MultivarAdapter
  - DockerAdapter
  - ... (add your own adapter)
- Automatic algorithm detection quality scoring using [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) (Area under the ROC curve, also _c-statistic_) metric
- Automatic timing of the algorithm execution (differentiates pre-, main-, and post-processing)
- Distributed experiment execution
- Output and logfile tracking for subsequent inspection

## Installation

TimeEval can be installed as a package or from source.

### Installation using `pip`

Builds of `TimeEval` are published to the [internal package registry](https://gitlab.hpi.de/akita/timeeval/-/packages) of the Gitlab instance running at [gitlab.hpi.de](https://gitlab.hpi.de/) and to [PyPI](https://pypi.org/project/TimeEval/).

#### Prerequisites

- python >= 3.7, <=3.9
- pip >= 20
- Docker
- (optional) A [personal access token](https://gitlab.hpi.de/help/user/profile/personal_access_tokens.md) with the scope set to `api` (read) or another type of access token able to read the package registry of TimeEval hosted at [gitlab.hpi.de](https://gitlab.hpi.de/).
- (optional) `rsync` for distributed TimeEval

#### Steps

You can use `pip` to install TimeEval from PyPI:

```sh
pip install TimeEval
```

### Installation from source

**tl;dr**

```bash
git clone git@gitlab.hpi.de:akita/bp2020fn1/timeeval.git
cd timeeval/
conda env create --file environment.yml
conda activate timeeval
python setup.py install
```

#### Prerequisites

The following tools are required to install TimeEval from source:

- git
- conda (anaconda or miniconda)

#### Steps

1. Clone this repository using git and change into its root directory.
2. Create a conda-environment and install all required dependencies.
   Use the file [`environment.yml`](./environment.yml) for this:
   `conda env create --file environment.yml`.
3. Activate the new environment and install TimeEval using _setup.py_:
   `python setup.py install`.
4. If you want to make changes to TimeEval or run the tests, you need to install the development dependencies from `requirements.dev`:
   `pip install -r requirements.dev`.

## Usage

**tl;dr**

```python
from typing import Dict, Any

import numpy as np

from timeeval import TimeEval, DatasetManager, Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import FunctionAdapter
from timeeval.constants import HPI_CLUSTER
from timeeval.params import FixedParameters


# Load dataset metadata
dm = DatasetManager(HPI_CLUSTER.akita_dataset_paths[HPI_CLUSTER.BENCHMARK], create_if_missing=False)

# Define algorithm
def my_algorithm(data: np.ndarray, args: Dict[str, Any]) -> np.ndarray:
    score_value = args.get("score_value", 0)
    return np.full_like(data, fill_value=score_value)

# Select datasets and algorithms
datasets = dm.select(collection="NAB")
datasets = datasets[-1:]
# Add algorithms to evaluate...
algorithms = [
    Algorithm(
        name="MyAlgorithm",
        main=FunctionAdapter(my_algorithm),
        data_as_file=False,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality.UNIVARIATE,
        param_config=FixedParameters({"score_value": 1.})
    )
]
timeeval = TimeEval(dm, datasets, algorithms)

# execute evaluation
timeeval.run()
# retrieve results
print(timeeval.get_results())
```

## ESA-ADB Citation

If you refer to ESA-ADB in your work, please cite our paper:

> Krzysztof Kotowski, Christoph Haskamp, Jacek Andrzejewski, Bogdan Ruszczak, Jakub Nalepa, Daniel Lakey, Peter Collins, Aybike Kolmas, Mauro Bartesaghi, Jose Martínez-Heras, and Gabriele De Canio.
> European Space Agency Benchmark for Anomaly Detection in Satellite Telemetry. arXiv, 2024.
> doi:[10.48550/arXiv.2406.17826](https://doi.org/10.48550/arXiv.2406.17826)

```bibtex
@article{kotowski_european_2024,
  title = {European {Space} {Agency} {Benchmark} for {Anomaly} {Detection} in {Satellite} {Telemetry}},
  author = {Kotowski, Krzysztof and Haskamp, Christoph and Andrzejewski, Jacek and Ruszczak, Bogdan and Nalepa, Jakub and Lakey, Daniel and Collins, Peter and Kolmas, Aybike and Bartesaghi, Mauro and Martinez-Heras, Jose and De Canio, Gabriele},
  date = {2024},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2406.17826}
}
```

## TimeEval Citation

If you use TimeEval in your project or research, please cite the demonstration paper:

> Phillip Wenig, Sebastian Schmidl, and Thorsten Papenbrock.
> TimeEval: A Benchmarking Toolkit for Time Series Anomaly Detection Algorithms. PVLDB, 15(12): 3678 - 3681, 2022.
> doi:[10.14778/3554821.3554873](https://doi.org/10.14778/3554821.3554873)

```bibtex
@article{WenigEtAl2022TimeEval,
  title = {TimeEval: {{A}} Benchmarking Toolkit for Time Series Anomaly Detection Algorithms},
  author = {Wenig, Phillip and Schmidl, Sebastian and Papenbrock, Thorsten},
  date = {2022},
  journaltitle = {Proceedings of the {{VLDB Endowment}} ({{PVLDB}})},
  volume = {15},
  number = {12},
  pages = {3678--3681},
  doi = {10.14778/3554821.3554873}
}
```
