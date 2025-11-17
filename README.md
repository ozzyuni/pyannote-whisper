# pyannote-whisper

Run ASR and speaker diarization based on whisper and pyannote.audio.

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

Same as whisper except a new param `diarization`:

    python -m pyannote_whisper.cli.transcribe data/afjiv.wav --model large-v3 --diarization True

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
## Python usage 2
please find more details in [this](https://gist.github.com/hbredin/049f2b629700bcea71324d2c1e7f8337) notebook.

```python
import whisper
from pyannote.audio import Pipeline
from pyannote.audio import Audio
from pyannote_whisper.utils import diarize_text
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token="your/token")
model = whisper.load_model("tiny.en")
diarization_result = pipeline("data/afjiv.wav")

from pyannote.audio import Audio
audio = Audio(sample_rate=16000, mono=True)
audio_file = "data/afjiv.wav"
for segment, _, speaker in diarization_result.itertracks(yield_label=True):
    waveform, sample_rate = audio.crop(audio_file, segment)
    text = model.transcribe(waveform.squeeze().numpy())["text"]
    print(f"{segment.start:.2f}s {segment.end:.2f}s {speaker}: {text}")

```
## Meeting summary from chatgpt

```
summarize the following words: SPEAKER_00  I think if you're a leader and you don't understand the terms that you're using, that's probably the first start.
SPEAKER_00  It's really important that as a leader in the organisation you understand what digitisation means.
SPEAKER_00  You take the time to read widely in the sector.
SPEAKER_00  There are a lot of really good books, Kevin Kelly, who started Wired magazine has written a great book on various technologies.
...
```
```
Leaders need to understand the terms and technologies of digitization in order to separate hype from hope and apply it effectively to their business. Becoming data-driven and collaborating with technology vendors can help organizations achieve their desired outcomes. Procurement processes need to be fundamentally rethought to improve functionality and eliminate unnecessary spending.
```

```
summarize the opinions of different speaker
```

```
Speaker 00 emphasizes the importance of leaders understanding digitization terms and technologies to apply them effectively to their business. 
Speaker 01 suggests that taking control of spend data is a crucial first step in digital transformation. 
Speaker 02 believes that procurement processes need to be rethought to improve functionality and eliminate unnecessary spending. 
Speaker 03 advises understanding the purpose of digitization and collaborating with colleagues for a better outcome. 
Speaker 04 suggests thinking big and being brave, as well as talking to technology vendors to achieve desired outcomes.
```
