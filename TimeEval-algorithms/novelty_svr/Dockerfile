FROM registry.gitlab.hpi.de/akita/i/python3-base

LABEL maintainer="sebastian.schmidl@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

# install python dependencies
COPY requirements.txt .
RUN pip install -r ./requirements.txt

COPY model.py .
COPY algorithm.py .
COPY manifest.json .
