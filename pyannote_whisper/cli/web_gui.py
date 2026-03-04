import datetime
import json
import os
import gradio as gr

from pyannote_whisper.cli.transcribe import parse_args, cli

CONFIG_FILE = "pyannote_whisper.json"

INSTRUCTIONS = """Upload audio file(s) to begin. If multiple files are uploaded, output  will be zipped.

The Web GUI includes a limited number of parameters to adjust:

  'Whisper model': Select which Whisper model to use, large-v3 and large-v3-turbo are generally recommended
  'Output format': The output of Whisper can be formatted as a plain text, or subtitles in SRT or  VTT format
  'Diarization': PyAnnote diarization can be disbaled e.g. for pure subtitle generation
"""
import zipfile

class PyAnnoteWhisperGUI:
  def __init__(self):
    self.interface = gr.Blocks()
    self.args = {'model': "large-v3",
                 'output_format': "TXT",
                 'diarization': True,
                 'audio': None}
    self.transcript_path = None
    self.diarization_path = None
    self.config_file = CONFIG_FILE
    self.last_log_msg = None # Prevents duplicate messages from strange gradio event behaviour
    self.log = ""

  def launch(self):

    whisper_models = ['large-v3', 'large-v3-turbo', 'medium', 'small']
    if os.path.exists(self.config_file):
      with open(self.config_file, "r") as f:
        custom_models = json.load(f)["whisper_models"]
        whisper_models.extend(custom_models)

    with self.interface:
      whisper = gr.Dropdown(label="Whisper model", choices=whisper_models)
      transcript_type = gr.Dropdown(label="Output format", choices=['txt', 'vtt', 'srt'])
      diarization = gr.Dropdown(label="Diarization", choices=['enabled', 'disabled']) 
      log_box = gr.Textbox(label="Output", placeholder=INSTRUCTIONS, lines=10)

      audio = gr.UploadButton(label="Upload audio file(s)", file_count='multiple')
      process = gr.Button(value="Start transcription")
      transcript_download = gr.DownloadButton(label="Download raw transcript(s)")
      diarization_download = gr.DownloadButton(label="Download diarized transcript(s)")

      whisper.input(fn=self.set_whisper, inputs=[whisper], outputs=[log_box])
      transcript_type.input(fn=self.set_output_format, inputs=[transcript_type], outputs=[log_box])
      audio.upload(fn=self.upload_audio, inputs=[audio], outputs=[log_box])
      diarization.input(fn=self.set_diarization, inputs=[diarization], outputs=[log_box])
      process.click(fn=self.process_file, outputs=[transcript_download, diarization_download, log_box])

      transcript_download.click(fn=self.download_transcript, inputs=[transcript_download], outputs=[log_box])
      diarization_download.click(fn=self.download_diarization, inputs=[diarization_download], outputs=[log_box])

    self.interface.launch()

  def add_to_log(self, msg: str):
    line = str(datetime.datetime.now().strftime("%c")) + f"  {msg}"

    if line != self.last_log_msg:
      self.last_log_msg = line
      self.log = (self.log + '\n' + line).strip()

  def set_whisper(self, model: str):
    self.args['model'] = model
    self.add_to_log(f"Changed whisper model: '{model}'")
    return self.log

  def set_output_format(self, format: str):
    self.args['output_format'] = format.upper()
    self.add_to_log(f"Changed output format: '{format}'")
    return self.log
  
  def set_diarization(self, diarization: str):
    self.args['diarization'] = True if diarization=="enabled" else False
    self.add_to_log(f"Changed diarization setting: {diarization}")
    return self.log

  def upload_audio(self, audio: list[str]):
    self.args['audio'] = audio

    self.add_to_log(f"Audio file(s) uploaded:\n{"\n".join(audio)}")
    return self.log

  def download_transcript(self, value: str | None):
    self.add_to_log(f"Attempting to download file: {self.transcript_path}")
    return self.log
  
  def download_diarization(self, value: str | None):
    self.add_to_log(f"Attempting to download file: {self.diarization_path}")
    return self.log

  def prepare_for_download(self, file_paths: list[str], zip_name: str) -> str | None:
    
    if len(file_paths) == 1:
      return file_paths[0]
    elif len(file_paths) > 1:
      path, filename = os.path.split(file_paths[1])
      zip_path = os.path.join(path, zip_name)

      with zipfile.ZipFile(zip_path, 'w') as zip_file:
        for file_path in file_paths:
          zip_file.write(file_path)
      return zip_path
    else:
      return None

  def process_file(self):
    args = parse_args(require_audio=False)

    if os.path.exists(self.config_file):
      with open(self.config_file, "r") as f:
        custom_args = json.load(f)["args"]
        args.update(custom_args)

    args.update(self.args)
    msg = ""
    transcript_paths, diarization_paths = [], []
    try:
      transcript_paths, diarization_paths = cli(args)
      msg = "Successfully processed audio"
    except Exception as e:
      msg = f"Failed to process data with error: {e}"
  
    datecode = datetime.datetime.now().strftime("%Y%m%d%H%S")
    self.transcript_path = self.prepare_for_download(transcript_paths, f"{datecode}_raw_transcripts.zip")
    self.diarization_path = self.prepare_for_download(diarization_paths, f"{datecode}_diarized_transcripts.zip")

    self.add_to_log(msg)
    return self.transcript_path, self.diarization_path, self.log

def main():
  gui = PyAnnoteWhisperGUI()
  gui.launch()

if __name__ == "__main__":
  main()