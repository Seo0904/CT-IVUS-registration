FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

LABEL maintainer="keisuke.kanda@human.ait.kyushu-u.ac.jp"

ENV DEBIAN_FRONTEND=noninteractive
ENV WORK_DIR=/workspace

RUN apt-get clean && rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata \
    bash curl fish git git-lfs nano sudo \
    software-properties-common \
    python3 python3-pip python3-venv python3-dev \
    libopencv-dev && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

ARG UID=1000
ARG USER=hoge
ARG PASSWORD=hoge

RUN groupadd -g 1000 ${USER}_group && \
    useradd -m --uid=${UID} --gid=${USER}_group --groups=sudo ${USER} && \
    echo ${USER}:${PASSWORD} | chpasswd && \
    echo 'root:root' | chpasswd && \
    mkdir ${WORK_DIR} && \
    chown ${USER}:${USER}_group ${WORK_DIR}

ENV PATH=${PATH}:/home/${USER}/.local/bin

WORKDIR ${WORK_DIR}
USER ${USER}

COPY requirements.txt /

RUN python3 -m pip install --upgrade pip wheel && \
    python3 -m pip install "setuptools==69.5.1" && \
    python3 -m pip install --no-build-isolation --no-cache-dir visdom==0.2.4 && \
    python3 -m pip install --no-cache-dir -r /requirements.txt