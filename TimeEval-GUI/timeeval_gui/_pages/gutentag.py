import warnings
from typing import Tuple, Dict, Union, Optional

import streamlit as st
from gutenTAG import GutenTAG
from gutenTAG.generator.timeseries import TrainingType

from timeeval_gui.timeseries_config import TimeSeriesConfig
from timeeval_gui.utils import get_base_oscillations, get_anomaly_types, get_anomaly_params, \
    get_base_oscillation_parameters
from .page import Page
from ..files import Files


def general_area(ts_config: TimeSeriesConfig) -> TimeSeriesConfig:
    ts_config.set_name(st.text_input("Name"))
    ts_config.set_length(st.number_input("Length", min_value=10, value=1000))

    if st.checkbox("Generate training time series for supervised methods"):
        ts_config.set_supervised()
    if st.checkbox("Generate training time series for semi-supervised methods"):
        ts_config.set_semi_supervised()
    return ts_config


def select_base_oscillation(key="base-oscillation") -> Tuple[str, str]:
    bos = get_base_oscillations()
    value = st.selectbox("Base-Oscillation", bos.items(), format_func=lambda x: x[1], key=key)
    return value


def select_anomaly_type(key: str, bo_kind: str) -> Tuple[str, str]:
    anomaly_types = get_anomaly_types(bo_kind)
    return st.selectbox("Anomaly Type", anomaly_types.items(), format_func=lambda x: x[1], key=key)


def base_oscillation_area(c, ts_config: Optional[TimeSeriesConfig], return_dict: bool = False) -> Union[TimeSeriesConfig, Dict]:
    key = f"base-oscillation-{c}"
    base_oscillation = select_base_oscillation(key)
    parameters = get_base_oscillation_parameters(base_oscillation[0])
    param_config = {}
    for p in parameters:
        if p.tpe == "number":
            param_config[p.key] = st.number_input(p.name, key=f"{p.key}-{c}", help=p.help)
        elif p.tpe == "integer":
            param_config[p.key] = int(st.number_input(p.name, key=f"{p.key}-{c}", help=p.help))
        elif p.tpe == "object" and p.key == "trend":
            if st.checkbox("add Trend", key=f"{key}-add-trend"):
                st.markdown("---")
                param_config[p.key] = base_oscillation_area(f"{key}-{p.name}", None, return_dict=True)
                st.markdown("---")
        else:
            warn_msg = f"Input type ({p.tpe}) for parameter {p.name} of BO {base_oscillation[1]} not supported yet!"
            warnings.warn(warn_msg)
            st.warning(warn_msg)

    if return_dict:
        param_config["kind"] = base_oscillation[0]
        return param_config

    ts_config.add_base_oscillation(base_oscillation[0], **param_config)

    return ts_config


def anomaly_area(a, ts_config: TimeSeriesConfig) -> TimeSeriesConfig:
    position = st.selectbox("Position", key=f"anomaly-position-{a}", options=["beginning", "middle", "end"], index=1)
    length = int(st.number_input("Length", key=f"anomaly-length-{a}", min_value=1))
    channel = st.selectbox("Channel", key=f"anomaly-channel-{a}",
                           options=list(range(len(ts_config.config["base-oscillations"]))))

    n_kinds = st.number_input("Number of Anomaly Types", key=f"anomaly-types-{a}", min_value=1)
    kinds = []
    for t in range(int(n_kinds)):
        st.write(f"##### Type {t}")
        bo_kind = ts_config.config["base-oscillations"][channel]["kind"]
        anomaly_type, _ = select_anomaly_type(f"anomaly-type-{a}-{t}", bo_kind)
        parameters = parameter_area(a, t, anomaly_type, bo_kind)
        kinds.append({"kind": anomaly_type, "parameters": parameters})

    ts_config.add_anomaly(position=position, length=length, channel=channel, kinds=kinds)
    return ts_config


def parameter_area(a, t, anomaly_type: str, bo_kind: str) -> Dict:
    param_conf = {}
    parameters = get_anomaly_params(anomaly_type)
    for name, p, desc in parameters:
        if name.lower() == "sinusoid_k" and bo_kind != "sine":
            continue
        if name.lower() == "cbf_pattern_factor" and bo_kind != "cylinder-bell-funnel":
            continue

        key = f"{a}-{t}-{name}"
        if p == str:
            param_conf[name] = st.text_input(name.upper(), key=key, help=desc)
        elif p == bool:
            param_conf[name] = st.checkbox(name.upper(), key=key, help=desc)
        elif p == int:
            param_conf[name] = st.number_input(name.upper(), key=key, step=1, help=desc)
        elif p == float:
            param_conf[name] = st.number_input(name.upper(), key=key, help=desc)
    return param_conf


class GutenTAGPage(Page):
    def _get_name(self) -> str:
        return "GutenTAG"

    def render(self):
        st.image("images/gutentag.png")

        timeseries_config = TimeSeriesConfig()

        st.write("## General Settings")
        timeseries_config = general_area(timeseries_config)

        st.write("## Channels")
        n_channels = st.number_input("Number of Channels", min_value=1)
        for c in range(n_channels):
            with st.expander(f"Channel {c}"):
                timeseries_config = base_oscillation_area(c, timeseries_config)

        st.write("## Anomalies")
        n_anomalies = st.number_input("Number of Anomalies", min_value=0)
        for a in range(n_anomalies):
            with st.expander(f"Anomaly {a}"):
                timeseries_config = anomaly_area(a, timeseries_config)

        st.write("---")

        gt = None
        if st.button("Build Timeseries"):
            if gt is None:
                gt = GutenTAG.from_dict({"timeseries": [timeseries_config.config]}, plot=False)
                gt.generate()

            ts = gt.timeseries[0]

            test_data = ts.to_dataframe(training_type=TrainingType.TEST)
            st.write("### Test Data")
            st.line_chart(data=test_data)

            if ts.semi_supervised:
                semi_supervised_data = ts.to_dataframe(training_type=TrainingType.TRAIN_NO_ANOMALIES)
                st.write("### Semi-Supervised Training Data (no anomalies)")
                st.line_chart(data=semi_supervised_data.iloc[:, :-1])

            if ts.supervised:
                supervised_data = ts.to_dataframe(training_type=TrainingType.TRAIN_ANOMALIES)
                st.write("### Supervised Training Data (with anomalies)")
                st.line_chart(data=supervised_data)

        if st.button("Save"):
            if gt is None:
                gt = GutenTAG.from_dict({"timeseries": [timeseries_config.config]}, plot=False)
                gt.generate()
            Files().store_ts(gt)
            st.success(f"> Successfully saved new time series dataset '{timeseries_config.config['name']}' to disk.")
