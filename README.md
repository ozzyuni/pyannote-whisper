# pyannote-whisper

Run ASR and speaker diarization based on whisper and pyannote.audio. Compared to the original, this fork updates `pyannote` to `speaker_diarization_community_1` and loads `whisper` through `transformers` for some different configuration options. It also provides a new Docker-based setup option.

## Installation
1. Install transformers and accelerate
2. Install pyannote.audio

Recommended: Use the provided Docker setups instead:

    docker compose -f docker-compose-cuda.yml build

## Command-line usage
Open a command line in Docker

    docker compose -f docker-compose-cuda.yml run pyannote_whisper bash

Source the venv and install this package:

    source setup.sh

Can be used similarly to the original `pyannote_whisper`:

    python -m pyannote_whisper.cli.transcribe data/afjiv.wav --model large-v3 --diarization True

There have been some changes to available parameters, check if needed:

    python -m pyannote_whisper.cli.transcribe --help

By default, uses `cuda` for both steps if supported, and `flash attention 2` to further speed up `whisper`.

## Python usage

Transcription can also be performed within Python: 

```python
from pyannote_whisper.whisper import Whisper
from pyannote.audio import Pipeline as PyAnnotePipeline
from pyannote_whisper.utils import diarize_text
pipeline = PyAnnotePipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    use_auth_token="your/token"
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Whisper({
    'model':  "large-v3"
    'device': device
}

asr_result = model.transcribe("data/afjiv.wav")
diarization_result = pipeline("data/afjiv.wav").speaker_diarization
final_result = diarize_text(asr_result, diarization_result)

for seg, spk, sent in final_result:
    line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}'
    print(line)
```

```
0.00 41.16 SPEAKER_00  I think if you're a leader and you don't understand the terms that you're using that's probably the first start it's really important that as a leader in the organization you understand what digitization means you take the time to read widely in the sector there are a lot of really good books Kevin Kelly who started wide magazine has written a great great book on on various technologies I think understanding the technologies understanding what's out there so that you can separate the hype from the hope is really an important first step and then making sure you understand the relevance of that for your function and how that fits into your business is the second step I think
41.16 78.86 SPEAKER_01  to two simple suggestions you know one one is I love the phrase brilliant at the basics right so you know how can you become brilliant at the basics but beyond that you know the fundamental thing I've seen which hasn't changed is so few organizations as a first step have truly taken control of their spend data you know as a key first step on a digital transformation taking ownership of data and that's not a decision to use one vendor over someone else that says we are going to be completely data-driven we're going to try and be as real-time as possible and we're going to be able to explain that data to
78.86 95.22 SPEAKER_03  anyone the way they want to see it understand why you're doing it and then And the second thing is reach out to suppliers in the market, talk to them, collaborate with them, you'll get a much better outcome.
95.22 108.28 SPEAKER_04  Think about what outcome you want at the end instead of thinking about the different processes and their software names, so e-sourcing being one of 20.
108.28 119.26 SPEAKER_04  think big and be brave I think and talk to technology vendors because rather than just sending them forms we won't bite you.
119.26 135.68 SPEAKER_02  I think we should fundamentally all of us rethink how procurement should be done and then start to define the functionality that we need and how we can make this work. What What we're doing today is absolutely wrong.
135.68 145.26 SPEAKER_02  We don't like it, procurement people don't like it, our colleagues don't like it, nobody wants it, and we're spending a huge amount of money for no reason.

```
