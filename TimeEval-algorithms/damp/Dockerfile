FROM registry.gitlab.hpi.de/akita/i/python3-base

LABEL maintainer="phillip.wenig@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY . /app/
RUN pip install -r requirements.txt
