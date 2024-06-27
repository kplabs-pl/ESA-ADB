import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Dict, Hashable, Any, Optional

import pandas as pd
import requests
import yaml
from gutenTAG import GutenTAG
from gutenTAG.addons.timeeval import TimeEvalAddOn
from timeeval import Datasets, DatasetManager

from timeeval_gui.config import GUTENTAG_CONFIG_SCHEMA_ANOMALY_KIND_URL, TIMEEVAL_FILES_PATH


class Files:
    _instance: Optional['Files'] = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Files, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if TIMEEVAL_FILES_PATH.is_absolute():
            self._files_path = TIMEEVAL_FILES_PATH
        else:
            self._files_path = (Path.cwd() / TIMEEVAL_FILES_PATH).absolute()
        self._files_path.mkdir(parents=True, exist_ok=True)
        self._anomaly_kind_schema_path = self._files_path / "cache" / "anomaly-kind.guten-tag-generation-config.schema.yaml"
        self._anomaly_kind_schema_path.parent.mkdir(exist_ok=True)
        self._ts_path = self._files_path / "timeseries"
        self._ts_path.mkdir(exist_ok=True)
        self._results_path = self._files_path / "results"
        self._results_path.mkdir(exist_ok=True)

    def anomaly_kind_configuration_schema(self) -> Dict[Hashable, Any]:
        # load parameter configuration only once
        if not self._anomaly_kind_schema_path.exists():
            self._load_anomaly_kind_configuration_schema()
        with self._anomaly_kind_schema_path.open("r") as fh:
            return yaml.load(fh, Loader=yaml.FullLoader)

    def store_ts(self, gt: GutenTAG) -> None:
        # process time series with TimeEvalAddOn to create dataset metadata
        with tempfile.TemporaryDirectory() as tmp_path:
            tmp_path = Path(tmp_path)
            TimeEvalAddOn().process(gt.overview, gt, Namespace(output_dir=tmp_path, no_save=False))
            df_index = pd.read_csv(tmp_path / "datasets.csv").set_index(["collection_name", "dataset_name"])

        # store index file (and potentially merge with existing beforehand)
        if (self._ts_path / "datasets.csv").exists():
            df_existing_index = pd.read_csv(self._ts_path / "datasets.csv").set_index(
                ["collection_name", "dataset_name"])
            df_index = pd.concat([df_existing_index[~df_existing_index.index.isin(df_index.index)], df_index])
        df_index.to_csv(self._ts_path / "datasets.csv")

        # save time series
        gt.save_timeseries(self._ts_path)

        # remove overview file (contains outdated information)
        (self._ts_path / "overview.yaml").unlink()

    def dmgr(self) -> Datasets:
        return DatasetManager(self._ts_path, create_if_missing=False)

    def results_folder(self) -> Path:
        return self._results_path

    def timeseries_folder(self) -> Path:
        return self._ts_path

    def _load_anomaly_kind_configuration_schema(self) -> None:
        result = requests.get(GUTENTAG_CONFIG_SCHEMA_ANOMALY_KIND_URL)
        with self._anomaly_kind_schema_path.open("w") as fh:
            fh.write(result.text)
