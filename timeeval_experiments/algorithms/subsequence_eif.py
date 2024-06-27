from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_subsequence_eif_parameters: Dict[str, Dict[str, Any]] = {
 "extension_level": {
  "defaultValue": None,
  "description": "Extension level 0 resembles standard isolation forest. If unspecified (`None`), then `extension_level=X.shape[1] - 1`.",
  "name": "extension_level",
  "type": "int"
 },
 "limit": {
  "defaultValue": None,
  "description": "The maximum allowed tree depth. This is by default set to average length of unsucessful search in a binary tree.",
  "name": "limit",
  "type": "int"
 },
 "max_samples": {
  "defaultValue": None,
  "description": "The number of samples to draw from X to train each base estimator: `max_samples * X.shape[0]`. If unspecified (`None`), then `max_samples=min(256, X.shape[0])`.",
  "name": "max_samples",
  "type": "float"
 },
 "n_trees": {
  "defaultValue": 200,
  "description": "The number of decision trees (base estimators) in the forest (ensemble).",
  "name": "n_trees",
  "type": "int"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
  "name": "random_state",
  "type": "int"
 },
    "window_size": {
        "defaultValue": 100,
        "description": "Size of the sliding windows to extract subsequences as input to LOF.",
        "name": "window_size",
        "type": "int"
    },
 "target_channels": {
    "defaultValue": None,
    "description": "channels to detect anomalies in. If None, all channels in data are used",
    "name": "target_channels",
    "type": "List[String]"
 },
}


def subsequence_eif(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="Subsequence EIF",
        main=DockerAdapter(
            image_name="registry.gitlab.hpi.de/akita/i/subsequence_eif",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_subsequence_eif_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
