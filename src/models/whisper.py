import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from src.models import BaseASR


class Whisper(BaseASR):
    def __init__(self, model_name="openai/whisper-small", chunk_length_s: int = 30):
        self.model_name = model_name
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=0 if torch.cuda.is_available() else -1,
            chunk_length_s=chunk_length_s,
            return_timestamps=True,
        )

    def generate(self, input, batch_size: int | None = None, **kwargs):
        if not isinstance(input, list):
            input = [input]
        bs = batch_size or len(input)
        outputs = self.pipe(input, batch_size=bs)
        if isinstance(outputs, dict):
            outputs = [outputs]
        return outputs


# if __name__=="__main__":
#     asr=load_asr("tiny")
#     out = asr("audio/001.wav")
