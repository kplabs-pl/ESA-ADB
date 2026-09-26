from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_struktura_dfa_parameters: Dict[str, Dict[str, Any]] = {
 "window_size": {
  "defaultValue": 1024,
  "description": "Samples per DFA window (at least 72).",
  "name": "window_size",
  "type": "Int"
 },
 "stride": {
  "defaultValue": 256,
  "description": "Step between consecutive windows.",
  "name": "stride",
  "type": "Int"
 },
 "baseline_fraction": {
  "defaultValue": 0.2,
  "description": "Fraction of the windows (from the start) used for the baseline.",
  "name": "baseline_fraction",
  "type": "Float"
 },
 "threshold": {
  "defaultValue": 3.5,
  "description": "Robust z-score of the window's DFA exponent above which the window is flagged.",
  "name": "threshold",
  "type": "Float"
 },
 "causal": {
  "defaultValue": False,
  "description": "If true, flag only the stride samples after each decision (no look-ahead); if false, flag the whole window.",
  "name": "causal",
  "type": "boolean"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Not used (deterministic). Present for framework compatibility.",
  "name": "random_state",
  "type": "Int"
 },
    "target_channels": {
        "defaultValue": None,
        "description": "channels to detect anomalies in. If None, all channels in data are used",
        "name": "target_channels",
        "type": "List[String]"
    },
}


def struktura_dfa(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="struktura_dfa",
        main=DockerAdapter(
            image_name="registry.gitlab.hpi.de/akita/i/struktura_dfa",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_struktura_dfa_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
