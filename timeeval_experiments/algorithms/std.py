from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_std_parameters: Dict[str, Dict[str, Any]] = {
 "tol": {
  "defaultValue": 3.0,
  "description": "Number of standard deviation above/below the mean to treat as an anomaly.",
  "name": "tol",
  "type": "Float"
 },
 "random_state": {
  "defaultValue": 42,
  "description": "Seed for random number generation.",
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


def std(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="std",
        main=DockerAdapter(
            image_name="registry.gitlab.hpi.de/akita/i/std",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_std_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
