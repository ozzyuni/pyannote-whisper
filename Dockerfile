FROM ubuntu:24.04

# User setup
RUN userdel -r ubuntu && \
    useradd -m pyannote_whisper && \
    passwd pyannote_whisper -d

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

RUN echo "pyannote_whisper ALL=(ALL:ALL) NOPASSWD: ALL" | tee /etc/sudoers.d/pyannote_whisper

USER pyannote_whisper

# Venv setup
RUN python3 -m venv /venv/pyannote_whisper_venv

# Python requirements
RUN . /venv/pyannote_whisper_venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir jupyterlab transformers accelerate pyannote.audio pydub ipywidgets

USER root

# Additional requirements
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    ffmpeg

RUN rm /etc/sudoers.d/pyannote_whisper

COPY ./setup.py /workspace/setup.py

USER pyannote_whisper

WORKDIR /workspace

CMD source /venv/pyannote_whisper_venv/bin/activate