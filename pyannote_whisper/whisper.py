import warnings
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class Whisper:

    def __init__(self, args):
        self.args = args

        self.device = self.args.pop("device")
        self.torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        enable_fa2 = self.args.pop("flash_attention_2")
        self.attn_implementation = "flash_attention_2" if enable_fa2 and "cuda" in self.device else "sdpa"

        print("Attention mechanism:", self.attn_implementation)
        
        model_name: str = self.args.pop("model")
        self.model_id = "openai/whisper-" + model_name
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation=self.attn_implementation,
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            dtype=self.torch_dtype,
            return_timestamps=True,
            device=self.device,
        )

    def transcribe(self, filename):

        if self.model_id.endswith(".en") and self.args['language'] not in {"en", "English"}:
            if self.args['language'] is not None:
                warnings.warn(
                    f"{self.model_id} is an English-only model but receipted '{self.args['language']}'; using English instead.")
            self.args['language'] = "en"
        
        result = self.pipe(filename, generate_kwargs=self.args)

        return result
