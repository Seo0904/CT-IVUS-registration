FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

LABEL maintainer="keisuke.kanda@human.ait.kyushu-u.ac.jp"

# Timezone setting
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

# 基本ツール
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash curl fish git nano sudo software-properties-common

# Git LFS
RUN apt-get update && apt-get install -y --no-install-recommends git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libopencv-dev

# Python 3.9 インストール（ppaを使って正しく入れる）
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.9 python3.9-distutils python3.9-venv

# python3 → python3.9 を指すようにシンボリックリンクを作成
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3


# pip インストール
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Add User & Group
ARG UID
ARG USER
ARG PASSWORD
RUN groupadd -g 1000 ${USER}_group && \
    useradd -m --uid=${UID} --gid=${USER}_group --groups=sudo ${USER} && \
    echo ${USER}:${PASSWORD} | chpasswd && \
    echo 'root:root' | chpasswd

ENV PATH=${PATH}:/home/${USER}/.local/bin

# 作業ディレクトリ
ENV WORK_DIR=/workspace
RUN mkdir ${WORK_DIR} && \
    chown ${USER}:${USER}_group ${WORK_DIR}
WORKDIR ${WORK_DIR}

# ユーザー切り替え
USER ${USER}

# Python パッケージのインストール
COPY requirements.txt /
COPY requirements.txt /

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --no-build-isolation --no-cache-dir visdom==0.2.4 && \
    python3 -m pip install --no-cache-dir -r /requirements.txt
