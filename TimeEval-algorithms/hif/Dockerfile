FROM registry.gitlab.hpi.de/akita/i/python3-base

LABEL maintainer="yannik.schroeder@student.hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

# install algorithm dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY hif.py /app/
COPY algorithm.py /app/
