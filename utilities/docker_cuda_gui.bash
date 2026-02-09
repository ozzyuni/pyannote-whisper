#!/usr/bin/env bash
xhost +local:docker
docker compose -f docker-compose-cuda-gui.yml run pyannote_whisper bash
docker compose -f docker-compose-cuda-gui.yml down --remove-orphans
