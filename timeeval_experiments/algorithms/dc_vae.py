from durations import Duration
from typing import Any, Dict, Optional

from timeeval import Algorithm, TrainingType, InputDimensionality
from timeeval.adapters import DockerAdapter
from timeeval.params import ParameterConfig


_dc_vae_parameters: Dict[str, Dict[str, Any]] = {
 "alpha": {
  "defaultValue": 3,
  "description": "Anomaly detection threshold of the number of standard deviations from mean",
  "name": "alpha",
  "type": "Float"
 },
 "max_std": {
  "defaultValue": 7,
  "description": "Threshold for removing outliers",
  "name": "max_std",
  "type": "Float"
 },
 "T": {
  "defaultValue": 128,
  "description": "Input window size",
  "name": "T",
  "type": "Int"
 },
 "cnn_units": {
  "defaultValue": [64, 64, 64, 64, 64, 64],
  "description": "Number of CNN units in each layer",
  "name": "cnn_units",
  "type": "List[Int]"
 },
 "dil_rate": {
  "defaultValue": [1, 2, 4, 8, 16, 32, 64],
  "description": "Dilation rates in consecutive layers",
  "name": "dil_rate",
  "type": "List[Int]"
 },
 "kernel": {
  "defaultValue": 2,
  "description": "Kernel size of CNN layers",
  "name": "kernel",
  "type": "Int"
 },
 "strs": {
  "defaultValue": 1,
  "description": "Stride length of CNN layers",
  "name": "lstm_batch_size",
  "type": "Int"
 },
 "batch_size": {
  "defaultValue": 64,
  "description": "Batch size for network training",
  "name": "batch_size",
  "type": "Int"
 },
 "J": {
  "defaultValue": 2,
  "description": "Dimensionality of the latent space",
  "name": "J",
  "type": "Int"
 },
 "epochs": {
  "defaultValue": 100,
  "description": "number of epochs to run",
  "name": "epochs",
  "type": "Int"
 },
 "lr": {
  "defaultValue": 1e-3,
  "description": "Learning rate",
  "name": "lr",
  "type": "Float"
 },
 "lr_decay": {
  "defaultValue": False,
  "description": "decide if use exponential lr decay",
  "name": "lr_decay",
  "type": "Bool"
 },
 "decay_rate": {
  "defaultValue": 0.96,
  "description": "Learning rate decay rate",
  "name": "decay_rate",
  "type": "Float"
 },
 "decay_step": {
  "defaultValue": 7000,
  "description": "Learning rate decay step",
  "name": "decay_step",
  "type": "Int"
 },
 "val_percent": {
  "defaultValue": 0.2,
  "description": "Percentage of data to use as a validation set",
  "name": "val_percent",
  "type": "Float"
 },
 "validation_date_split": {
  "defaultValue": None,
  "description": "The date (a string compatible with Python datetime format) at which to split the training dataset"
                 "into training and validation subsets. If None, the data are split randomly according to 'split' param.",
  "name": "validation_date_split",
  "type": "String"
 },
 "seed": {
  "defaultValue": 123,
  "description": "Random seed",
  "name": "seed",
  "type": "Int"
 },
 "input_channels": {
  "defaultValue": None,
  "description": "channels to detect anomalies in. If None, all channels in data are used",
  "name": "input_channels",
  "type": "List[String]"
 },
 "target_channels": {
  "defaultValue": None,
  "description": "channels to detect anomalies in. If None, all channels in data are used",
  "name": "target_channels",
  "type": "List[String]"
 },
 "threshold_scores": {
  "defaultValue": False,
  "description": "if True, threshold anomaly scores using Telemanom dynamic thresholding",
  "name": "threshold_scores",
  "type": "Bool"
 }
}


def dc_vae(params: ParameterConfig = None, skip_pull: bool = False, timeout: Optional[Duration] = None) -> Algorithm:
    return Algorithm(
        name="DC-VAE",
        main=DockerAdapter(
            image_name="registry.gitlab.hpi.de/akita/i/dc_vae",
            skip_pull=skip_pull,
            timeout=timeout,
            group_privileges="akita",
        ),
        preprocess=None,
        postprocess=None,
        param_schema=_dc_vae_parameters,
        param_config=params or ParameterConfig.defaults(),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality("multivariate")
    )
