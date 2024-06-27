from abc import ABC, abstractmethod
import streamlit as st


class Page(ABC):
    def __init__(self):
        super().__init__()
        st.set_page_config(page_title=f"{self.name} | TimeEval - A Time Series Anomaly Detection Toolkit")

    @property
    def name(self) -> str:
        return self._get_name()

    @abstractmethod
    def _get_name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def render(self):
        raise NotImplementedError()
