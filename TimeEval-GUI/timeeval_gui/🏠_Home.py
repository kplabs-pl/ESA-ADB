import streamlit as st


def main():
    st.markdown("""
    # Welcome to the TimeEval GUI
    
    TimeEval includes an extensive data generator and supports both interactive and batch evaluation scenarios.
    This novel toolkit, aims to ease the evaluation effort and help the community to provide more meaningful evaluations
    in the Time Series Anomaly Detection field.
    
    This Tool has 3 main components:
    
    1. [GutenTAG](/GutenTAG) to generate time series 
    2. [Eval](/Eval) to run multiple anomaly detectors on multiple datasets
    3. [Results](/Results) to compare the quality of multiple anomaly detectors
    """)

    st.info("For more detailed documentation on the tools: "
            "[GutenTAG Documentation](https://github.com/HPI-Information-Systems/gutentag/blob/main/doc/index.md) and "
            "[Eval Documentation](https://timeeval.readthedocs.io/en/latest/)")


if __name__ == '__main__':
    main()
