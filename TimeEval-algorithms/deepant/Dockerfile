FROM registry.gitlab.hpi.de/akita/i/python3-torch

LABEL maintainer="grit.fessel@student.hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

# install algorithm dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY deepant /app/deepant
COPY algorithm.py /app/
COPY helper.py /app/
