#!/usr/bin/env bash

. /venv/pyannote_whisper_venv/bin/activate
pip install /pyannote_whisper
pyannote-whisper-gui
