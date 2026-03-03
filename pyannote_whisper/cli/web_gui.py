import gradio as gr
import datetime

from pyannote_whisper.cli.transcribe import parse_args, cli

INSTRUCTIONS="""Upload an audio file to begin.

The Web GUI includes a limited number of parameters to adjust:

  'Whisper model': Select which Whisper model to use, large-v3 and large-v3-turbo are generally recommended
  'Output format': The output of Whisper can be formatted as a plain text, or subtitles in SRT or  VTT format
  'Diarization': PyAnnote diarization can be disbaled e.g. for pure subtitle generation
"""

class PyAnnoteWhisperGUI:
  def __init__(self):
    self.interface = gr.Blocks()
    self.args = {'model': "large-v3",
                 'output_format': "TXT",
                 'diarization': True,
                 'audio': None}
    self.transcript_path = None
    self.diarization_path = None
    self.log = ""

  def launch(self):
    with self.interface:
      whisper = gr.Dropdown(label="Whisper model", choices=['large-v3', 'large-v3-turbo', 'medium', 'small'])
      transcript_type = gr.Dropdown(label="Output format", choices=['txt', 'vtt', 'srt'])
      diarization = gr.Dropdown(label="Diarization", choices=['enabled', 'disabled']) 
      log_box = gr.Textbox(label="Output", placeholder=INSTRUCTIONS, lines=10)

      audio = gr.UploadButton(label="Upload audio")
      process = gr.Button(value="Start transcription")
      transcript_download = gr.DownloadButton(label="Download raw transcript")
      diarization_download = gr.DownloadButton(label="Download diarized transcript")

      whisper.input(fn=self.set_whisper, inputs=[whisper], outputs=[log_box])
      transcript_type.input(fn=self.set_output_format, inputs=[transcript_type], outputs=[log_box])
      audio.upload(fn=self.upload_audio, inputs=[audio], outputs=[log_box])
      diarization.input(fn=self.set_diarization, inputs=[diarization], outputs=[log_box])
      process.click(fn=self.process_file, outputs=[log_box])

      transcript_download.click(fn=self.download_transcript, inputs=[transcript_download], outputs=[transcript_download])
      diarization_download.click(fn=self.download_diarization, inputs=[diarization_download], outputs=[diarization_download])

    self.interface.launch()

  def add_to_log(self, msg: str):
    line = str(datetime.datetime.now().strftime("%c")) + f"  {msg}"

    if not self.log.endswith(line):
      self.log = (self.log + '\n' + line).strip()

  def set_whisper(self, model: str):
    self.args['model'] = model
    self.add_to_log(f"Model choice updated: '{model}'")
    return self.log

  def set_output_format(self, format: str):
    self.args['output_format'] = format.upper()
    self.add_to_log(f"Changed output format: '{format}'")
    return self.log
  
  def set_diarization(self, diarization: str):
    self.args['diarization'] = True if "enabled" else False
    self.add_to_log(f"Changed diarization setting: {diarization}")
    return self.log

  def upload_audio(self, audio: str):
    self.args['audio'] = [audio]

    self.add_to_log(f"Audio file uploaded: {audio}")
    return self.log

  def download_transcript(self, value: str | None):
    return self.transcript_path
  
  def download_diarization(self, value: str | None):
    return self.diarization_path

  def process_file(self):
    args = parse_args(require_audio=False)
    args.update(self.args)
    msg = ""
    try:
      self.transcript_path, self.diarization_path = cli(args)
      msg = "Successfully processed audio"
    except Exception as e:
      msg = f"Failed to process data with error: {e}"
  
    self.add_to_log(msg)
    return self.log

def main():
  gui = PyAnnoteWhisperGUI()
  gui.launch()

if __name__ == "__main__":
  main()