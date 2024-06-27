FROM registry.gitlab.hpi.de/akita/i/python3-base

LABEL maintainer="sebastian.schmidl@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY dwt_mlead.py .
COPY algorithm.py .
COPY manifest.json .
