import tkinter as tk
from tkinter.filedialog import askopenfilename
import tkinter.scrolledtext as st

import argparse
import gc
import os
import time
from typing import Literal

import numpy as np
import torch

import threading

from pyannote_whisper.whisper import Whisper
from pyannote_whisper.whisper_utils import (WriteSRT, WriteTXT, WriteVTT, optional_float,
                           optional_int, str2bool, LANGUAGES, TO_LANGUAGE_CODE)
from pyannote_whisper.utils import diarize_text, write_to_txt

def process(audio_path: str, whisper_model: str, diarization_enabled: bool, logger_fn = print):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default=whisper_model, help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device to use for PyTorch inference")
    parser.add_argument("--flash_attention_2", type=str2bool, default=True,
                        help="whether to use Flash Attention 2 instead of the default SDPA attention, only applicable if device=cuda")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"],
                        help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None,
                        choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
                        help="language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--temperature", type=optional_float, default=0, help="temperature to use for sampling")                   
    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2,
                        help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4,
                        help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0,
                        help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6,
                        help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--threads", type=optional_int, default=0,
                        help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
    parser.add_argument("--diarization", type=str2bool, default=diarization_enabled,
                        help="whether to perform speaker diarization; True by default")

    parser.add_argument("--exclusive", type=str2bool, default=True,
                        help="whether to use the exclusive or overlapping results from speech diarization")
    
    parser.add_argument("--output_format", type=str, default="TXT", choices=['TXT', 'VTT', 'SRT'],
                        help="output format; TXT by default")

    args = parser.parse_args().__dict__
    output_dir: str = args.pop("output_dir")
    os.makedirs(output_dir, exist_ok=True)

    output_format: Literal['TXT', 'VTT', 'SRT'] = args.pop("output_format")

    temperature = float(args.pop("temperature"))
    temperature_increment_on_fallback = args.pop("temperature_increment_on_fallback")
    if temperature_increment_on_fallback is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    args.update({"temperature": temperature})

    device = args.get("device")

    threads = args.pop("threads")
    if threads > 0:
        torch.set_num_threads(threads)

    model_dir: str = args.pop("model_dir")

    if model_dir is not None:
        os.environ['HF_HOME'] = model_dir

    whisper = Whisper(args)

    diarization = args.pop("diarization")
    exclusive_mode = args.pop("exclusive")

    hf_token = os.environ.get('HF_TOKEN', None)

    if diarization and hf_token is not None:
        from pyannote.audio import Pipeline as PyAnnotePipeline
        pyannote_pipeline = PyAnnotePipeline.from_pretrained("pyannote/speaker-diarization-community-1",
                                            token=hf_token)
        
        pyannote_pipeline.to(torch.device(device))
        # create huggingface.co free account and create your access token ^ with access to read repos
        # also you will need to apply access forms for certain repos to get access to them (it's free too)
        # you will see which repos requires this additional actions as access errors when try to use the program 
    elif hf_token is None:
        logger_fn("No valid token found in the HF_TOKEN environment variable, disabling diarization")
        diarization = False


    logger_fn("Starting transcription")
    transctiption_start = time.time()

    result = whisper.transcribe(audio_path)

    transctiption_end = time.time()

    logger_fn("Transcription completed in" +
        str(round(transctiption_end - transctiption_start, 1)) +
        "seconds"
        )

    audio_basename = os.path.basename(audio_path)

    audio_basename = os.path.basename(audio_path)

    # Delete whisper from memory to clear up space on systems with limited (V)RAM
    del whisper
    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()

    if output_format == "TXT":
        # save TXT
        with open(os.path.join(output_dir, audio_basename + ".txt"), "w", encoding="utf-8") as file:
            WriteTXT(output_dir).write_result(result, file=file)

    elif output_format == "VTT":
        # save VTT
        with open(os.path.join(output_dir, audio_basename + ".vtt"), "w", encoding="utf-8") as file:
            WriteVTT(output_dir).write_result(result, file=file)

    elif output_format == "SRT":
        # save SRT
      with open(os.path.join(output_dir, audio_basename + ".srt"), "w", encoding="utf-8") as file:
            WriteSRT(output_dir).write_result(result, file=file)

    if diarization:
        logger_fn("Starting diarization")
        diarization_start = time.time()

        if exclusive_mode:
            diarization_result = pyannote_pipeline(audio_path).exclusive_speaker_diarization
        else:
            diarization_result = pyannote_pipeline(audio_path).speaker_diarization

        diarization_end = time.time()

        logger_fn("Diarization completed in",
            round(diarization_end - diarization_start, 1),
            "seconds"
            )

        filepath = os.path.join(output_dir, audio_basename + "_spk.txt")
        res = diarize_text(result, diarization_result)
        write_to_txt(res, filepath)

class PyAnnoteWhisperGUI:
  """
  Quick and dirty GUI using Tkinter. Don't look too closely!
  """
  def __init__(self):
    self.window = tk.Tk()
    self.window.title("PyAnnote-Whisper")
    
    self.window.rowconfigure(0, minsize=800, weight=1)
    self.window.columnconfigure(1, minsize=800, weight=1)

    self.text_box = st.ScrolledText(self.window)

    self.frm_buttons = tk.Frame(self.window, relief=tk.RAISED, bd=2)
    self.btn_select = tk.Button(self.frm_buttons, text="Open File", command=self.select_file)
    self.btn_start = tk.Button(self.frm_buttons, text="Start Processing", command=self.process_file)

    self.select_model_label = tk.Label(self.frm_buttons,  text='Whisper:', width=15)

    self.model_choices = ['small', 'medium', 'large-v3-turbo', 'large-v3']
    self.model_choice = tk.StringVar(self.window)
    self.model_choice.set('large-v3')
    self.select_model = tk.OptionMenu(self.frm_buttons, self.model_choice, *self.model_choices)

    self.pyannote_enabled_label = tk.Label(self.frm_buttons,  text='Diarization:', width=15)

    self.pyannote_enabled_choices = [True, False]
    self.pyannote_enabled_choice = tk.BooleanVar(self.window)
    self.pyannote_enabled_choice.set(True)
    self.pyannote_enabled = tk.OptionMenu(self.frm_buttons, self.pyannote_enabled_choice, *self.pyannote_enabled_choices)

    self.btn_select.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
    self.btn_start.grid(row=1, column=0, sticky="ew", padx=5)

    self.select_model_label.grid(row=3, column=0, sticky="ew", padx=5)
    self.select_model.grid(row=4, column=0, sticky="ew", padx=5)

    self.pyannote_enabled_label.grid(row=5, column=0, sticky="ew", padx=5)
    self.pyannote_enabled.grid(row=6, column=0, sticky="ew", padx=5)

    self.frm_buttons.grid(row=0, column=0, sticky="ns")
    self.text_box.grid(row=0, column=1, sticky="nsew")

    self.audio_file = None
    self.process_thread = None
    self.process_ongoing = threading.Event()
    self.process_lock = threading.Lock()
     
  def select_file(self):
    """Open a file for editing."""
    filepath = askopenfilename(
      filetypes=[("Audio Files", "*.wav"), ("All Files", "*.*")]
    )
    if not filepath:
      return
    
    self.audio_file = filepath

    self.text_box.delete("1.0", tk.END)

    self.text_box.insert(tk.END, "Loaded file: " + filepath)
    self.window.title(f"PyAnnote-Whisper - {filepath}")

  def process_file(self):

    self.process_lock.acquire()

    if self.process_ongoing.is_set():
      self.write_to_log("Can't process more than one audio file at a time!")
      self.process_lock.release()
      return
    else:
      self.process_ongoing.set()
      self.process_lock.release()

    if self.audio_file is not None:
      self.process_thread = threading.Thread(target=process, args=(
        self.audio_file, self.model_choice.get(),
        self.pyannote_enabled_choice.get(),
        self.write_to_log)
      )
      
      self.process_thread.start()

    else:
      self.text_box.insert(tk.INSERT, "No audio file selected!")

  def write_to_log(self, msg: str):
     self.text_box.insert(tk.INSERT, '\n' + msg)

  def run(self):
    self.window.mainloop() 

def main():
  gui = PyAnnoteWhisperGUI()
  gui.run()

if __name__ == "__main__":
  main()