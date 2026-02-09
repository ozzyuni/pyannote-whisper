#!/usr/bin/env bash
docker compose -f docker-compose-cuda.yml run pyannote_whisper bash
docker compose -f docker-compose-cuda.yml down --remove-orphans
