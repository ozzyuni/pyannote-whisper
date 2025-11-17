FROM ubuntu:24.04

# User setup
RUN userdel -r ubuntu && \
    useradd -m pyannote-whisper && \
    passwd pyannote-whisper -d

# Base requirements
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    sudo \
    curl \
    software-properties-common \
    wget \
    build-essential \ 
    cmake \
    git \
    python3-pip \
    python3-venv

# Workspace setup
RUN mkdir -p /workspace/src/util_scripts && chmod 777 /workspace && \
    mkdir /venv && chmod 777 /venv

RUN echo "pyannote-whisper ALL=(ALL:ALL) NOPASSWD: ALL" | tee /etc/sudoers.d/pyannote-whisper

USER pyannote-whisper

# Venv setup
RUN python3 -m venv /venv/pyannote-whisper_venv

# Python requirements
RUN . /venv/pyannote_whisper_venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir jupyterlab transformers accelerate pyannote.audio pydub ipywidgets

USER root

# Additional requirements
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    ffmpeg

RUN rm /etc/sudoers.d/pyannote-whisper

COPY ./setup.py /workspace/setup.py

USER pyannote-whisper

WORKDIR /workspace

CMD source /venv/pyannote-whisper_venv/bin/activate