from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, Optional

import numpy as np
import streamlit as st
from streamlit.state import NoValue


class InputParam:
    def __init__(self, algorithm, selected, optim: bool, key=""):
        self.algorithm = algorithm
        self.selected = selected
        self.optim = optim
        self.key = key

    def get_default_value(self, cc) -> Any:
        v = self.algorithm.param_schema[self.selected]["defaultValue"]
        if v is None:
            return NoValue()
        else:
            return cc(v)

    @abstractmethod
    def _render_optim(self, help: str) -> Any:
        ...

    @abstractmethod
    def _render_fixed(self, help: str) -> Any:
        ...

    def render(self) -> Any:
        help = self.algorithm.param_schema[self.selected]["description"]

        if self.optim:
            return self._render_optim(help)
        else:
            return self._render_fixed(help)

    @staticmethod
    def from_type(algorithm, selected, optim: bool, key="") -> Optional[InputParam]:
        tpe = algorithm.param_schema[selected]["type"]
        if tpe.lower() == "int":
            cl = IntegerInputParam
        elif tpe.lower() == "float":
            cl = FloatInputParam
        elif tpe.lower().startswith("bool"):
            cl = BoolInputParam
        elif tpe.lower().startswith("enum"):
            cl = EnumInputParam
        elif tpe.lower().startswith("list") and optim:
            st.error("A list parameter cannot be optimized yet")
            return None
        else:
            cl = StringInputParam

        return cl(algorithm, selected, optim, key)


class IntegerInputParam(InputParam):
    def _render_fixed(self, help: str) -> Any:
        return int(st.number_input("Value",
                                   value=self.get_default_value(int),
                                   help=help,
                                   step=1,
                                   key=self.key))

    def _render_optim(self, help: str) -> Any:
        col1, col2 = st.columns(2)
        with col1:
            start_value = int(st.number_input("Start Value",
                                              value=self.get_default_value(int),
                                              help=help,
                                              step=1,
                                              key=f"{self.key}-start"))
        with col2:
            end_value = int(st.number_input("End Value",
                                            value=self.get_default_value(int),
                                            step=1,
                                            key=f"{self.key}-end"))
        if start_value > end_value:
            st.error("Start value must be smaller or equal to end value")

        return list(range(start_value, end_value + 1))


class FloatInputParam(InputParam):
    def _render_fixed(self, help: str) -> Any:
        return st.number_input("Value",
                                value=self.get_default_value(float),
                                help=help,
                                step=None,
                                format="%f",
                                key=self.key)

    def _render_optim(self, help: str) -> Any:
        col1, col2, col3 = st.columns((2,2,1))
        with col1:
            start_value = st.number_input("Start value",
                                          value=self.get_default_value(float),
                                          help=help,
                                          step=None,
                                          format="%f",
                                          key=f"{self.key}-start")
        with col2:
            end_value = st.number_input("End value",
                                        value=self.get_default_value(float),
                                        help=help,
                                        step=None,
                                        format="%f",
                                        key=f"{self.key}-end")
        with col3:
            number_steps = int(st.number_input("Steps",
                                               value=2,
                                               step=1,
                                               key=f"{self.key}-steps"))
        return np.linspace(start_value, end_value, number_steps).tolist()


class BoolInputParam(InputParam):
    def _render_optim(self, help: str) -> Any:
        return st.multiselect("Values",
                              options=[True, False],
                              help=help,
                              key=self.key)

    def _render_fixed(self, help: str) -> Any:
        st.markdown("Value")
        return st.checkbox("",
                           value=self.get_default_value(bool),
                           help=help,
                           key=self.key)


def parse_enum_param_type(tpe: str) -> List[str]:
    option_str = tpe.split("[")[1].split("]")[0]
    return option_str.split(",")


class EnumInputParam(InputParam):
    def _render_enum(self, help: str, input_field_class, with_index=True) -> Any:
        default_value = self.algorithm.param_schema[self.selected]["defaultValue"]
        tpe = self.algorithm.param_schema[self.selected]["type"]

        try:
            default_index = parse_enum_param_type(tpe).index(default_value)
        except ValueError:
            default_index = 0

        kwargs = {}
        if with_index:
            kwargs["index"] = default_index

        return input_field_class("Value",
                            options=parse_enum_param_type(tpe),
                            help=help,
                            key=self.key, **kwargs)

    def _render_optim(self, help: str) -> Any:
        return self._render_enum(help, st.multiselect, with_index=False)

    def _render_fixed(self, help: str) -> Any:
        return self._render_enum(help, st.selectbox)


class StringInputParam(InputParam):
    def _render_optim(self, help: str) -> Any:
        value = st.text_input("Value (comma seperated)",
                             value=self.algorithm.param_schema[self.selected]["defaultValue"],
                             help=help,
                             key=self.key)
        return list(map(lambda x: x.strip(), value.split(",")))

    def _render_fixed(self, help: str) -> Any:
        return st.text_input("Value",
                             value=self.algorithm.param_schema[self.selected]["defaultValue"],
                             help=help,
                             key=self.key)
