FROM registry.gitlab.hpi.de/akita/i/python3-base

LABEL maintainer="rohan.sawahn@student.hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY algorithm.py /app/
