FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

LABEL maintainer="phillip.wenig@hpi.de"

RUN set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
	apt-get install setpriv; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \

VOLUME ["/data", "/results"]

WORKDIR /app

COPY entrypoint.sh /entrypoint.sh

RUN pip install --upgrade pip

COPY requirements.txt /tmp/
# install requirements and cleanup afterwards (also removes tests and cached cython files of the dependencies)
RUN set -eux; \
    pip install --no-cache-dir -r /tmp/requirements.txt; \
    find /usr/local -depth \
		\( \
			\( -type d -a \( -name tests -o -name idle_test \) \) \
			-o \
			\( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
		\) -exec rm -rf '{}' +; \
    rm -rf /tmp/* /var/tmp/* ~/.cache/pip


ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "execute-algorithm" ]

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY lstm_ad /app/lstm_ad
COPY algorithm.py /app/
