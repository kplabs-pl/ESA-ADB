from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_copod_parameters: Dict[str, Dict[str, Any]] = {
     "random_state": {
      "defaultValue": 42,
      "description": "Seed for random number generation.",
      "name": "random_state",
      "type": "int"
     },
    "target_channels": {
        "defaultValue": None,
        "description": "channels to detect anomalies in. If None, all channels in data are used",
        "name": "target_channels",
        "type": "List[String]"
    },
}


def copod(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="COPOD",
        main=DockerAdapter(
            image_name="registry.gitlab.hpi.de/akita/i/copod",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_copod_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
