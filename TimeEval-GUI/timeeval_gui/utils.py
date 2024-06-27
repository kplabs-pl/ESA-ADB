from dataclasses import dataclass
from typing import Dict, Tuple, List, Type

from timeeval_gui.files import Files


def get_base_oscillations() -> Dict[str, str]:
    return {
        "sine": "Sine",
        "random-walk": "Random Walk",
        "ecg": "ECG",
        "polynomial": "Polynomial",
        "cylinder-bell-funnel": "Cylinder Bell Funnel",
        "random-mode-jump": "Random Mode Jump",
        # "formula": "Formula"
    }


@dataclass
class BOParameter:
    key: str
    name: str
    tpe: str
    help: str


def get_base_oscillation_parameters(bo: str) -> List[BOParameter]:
    common = [
        BOParameter(key="variance", name="Variance", tpe="number", help="Noise factor dependent on amplitude"),
        BOParameter(key="trend", name="Trend", tpe="object",
                    help="Defines another base oscillation as trend that gets added to its parent object. "
                         "Can be recursively used!"),
        BOParameter(key="offset", name="Offset", tpe="number", help="Gets added to the generated time series"),
    ]
    return {
               "sine": [
                   BOParameter(key="frequency", name="Frequency", tpe="number",
                               help="Number of sine waves per 100 points"),
                   BOParameter(key="amplitude", name="Amplitude", tpe="number", help="+/- deviation from 0"),
                   BOParameter(key="freq-mod", name="Frequency modulation", tpe="number",
                               help="Factor (of base frequency) of the frequency modulation that changes the amplitude of the "
                                    "sine wave over time. The carrier wave always has an amplitude of 1.")
               ],
               "random-walk": [
                   BOParameter(key="amplitude", name="Amplitude", tpe="number", help="+/- deviation from 0"),
                   BOParameter(key="smoothing", name="Smoothing factor", tpe="number",
                               help="Smoothing factor for convolution dependent on length")
               ],
               "cylinder-bell-funnel": [
                   BOParameter(key="avg-pattern-length", name="Average pattern length", tpe="integer",
                               help="Average length of pattern in time series"),
                   BOParameter(key="amplitude", name="Amplitude", tpe="number",
                               help="Average amplitude of pattern in time series"),
                   BOParameter(key="variance-pattern-length", name="Variance pattern length", tpe="number",
                               help="Variance of pattern length in time series"),
                   BOParameter(key="variance-amplitude", name="Variance amplitude", tpe="number",
                               help="Variance of amplitude of pattern in time series"),
               ],
               "ecg": [
                   BOParameter(key="frequency", name="Frequency", tpe="number",
                               help="Number of hear beats per 100 points")
               ],
               "polynomial": [
                   BOParameter(key="polynomial", name="Polynomial parameters", tpe="list[number]",
                               help="See numpy documentation: https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.Polynomial.html#numpy.polynomial.polynomial.Polynomial")
               ],
               "random-mode-jump": [
                   BOParameter(key="frequency", name="Frequency", tpe="number",
                               help="Number of jumps in Time Series"),
                   BOParameter(key="channel_diff", name="Channel mode difference", tpe="number",
                               help="Value difference of absolute mode values between channels"),
                   BOParameter(key="channel_offset", name="Channel offset", tpe="number",
                               help="Value offset from 0 in both directions"),
                   BOParameter(key="random_seed", name="Random seed", tpe="integer",
                               help="Random seed to have the similar channels"),
               ]
           }.get(bo, []) + common


def get_anomaly_types(bo_kind: str) -> Dict[str, str]:
    name_mapping = {
        "amplitude": "Amplitude",
        "extremum": "Extremum",
        "frequency": "Frequency",
        "mean": "Mean",
        "pattern": "Pattern",
        "pattern-shift": "Pattern Shift",
        "platform": "Platform",
        "trend": "Trend",
        "variance": "Variance",
        "mode-correlation": "Mode Correlation",
    }
    supported_anomalies = {
        "sine": ["amplitude", "extremum", "frequency", "mean", "pattern", "pattern-shift", "platform", "trend",
                 "variance"],
        "random-walk": ["amplitude", "extremum", "mean", "platform", "trend", "variance"],
        "ecg": ["amplitude", "extremum", "frequency", "mean", "pattern", "pattern-shift", "platform", "trend",
                "variance"],
        "polynomial": ["extremum", "mean", "platform", "trend", "variance"],
        "cylinder-bell-funnel": ["amplitude", "extremum", "mean", "pattern", "platform", "trend", "variance"],
        "random-mode-jump": ["mode-correlation"],
        "formula": ["extremum"]
    }
    return dict(map(lambda x: (x, name_mapping[x]), supported_anomalies.get(bo_kind, [])))


def map_types(t: str) -> Type:
    return {
        "boolean": bool,
        "string": str,
        "integer": int,
        "number": float
    }.get(t, str)


def get_anomaly_params(anomaly: str) -> List[Tuple[str, Type, str]]:
    params = []
    param_config = Files().anomaly_kind_configuration_schema()

    for param_name, param in param_config["definitions"].get(f"{anomaly}-params", {}).get("properties", {}).items():
        params.append((param_name, map_types(param.get("type")), param.get("description", "")))

    return params
