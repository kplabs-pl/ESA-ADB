# Separate build image!
#----------------------------
FROM registry.gitlab.hpi.de/akita/i/python3-base as build

LABEL maintainer="yannik.schroeder@student.hpi.de"

# install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

RUN python -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# build and install dependencies
RUN pip install cython
COPY requirements.txt .
RUN pip install -r ./requirements.txt
#----------------------------

FROM registry.gitlab.hpi.de/akita/i/python3-base as runtime

LABEL maintainer="yannik.schroeder@student.hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=build /opt/venv /opt/venv
COPY median_method.py /app/
COPY algorithm.py /app/
