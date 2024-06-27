FROM tensorflow/tensorflow:2.5.1-gpu-jupyter

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
	apt-get install setpriv; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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

COPY telemanom /app/telemanom
COPY algorithm.py /app/
