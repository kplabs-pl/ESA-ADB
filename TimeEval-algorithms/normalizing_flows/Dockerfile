FROM registry.gitlab.hpi.de/akita/i/python3-torch

LABEL maintainer="phillip.wenig@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY . /app
