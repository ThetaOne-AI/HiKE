import io
import os
import tempfile
import wave
from typing import Any, List

import numpy as np
from openai import OpenAI

from src.models import BaseASR


class GPTTranscribe(BaseASR):
    def __init__(
        self, model_name: str = "gpt-4o-transcribe", api_key: str | None = None
    ):
        self.client = OpenAI(api_key=(api_key or os.getenv("OPENAI_API_KEY")))
        self.model = model_name

    def _write_wav_bytes(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=-1)
        if np.issubdtype(audio_array.dtype, np.floating):
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_int16 = (audio_array * 32767.0).astype(np.int16)
        elif audio_array.dtype != np.int16:
            audio_int16 = audio_array.astype(np.int16)
        else:
            audio_int16 = audio_array

        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit PCM
                wav_file.setframerate(int(sample_rate))
                wav_file.writeframes(audio_int16.tobytes())
            return buffer.getvalue()

    def _normalize_inputs(self, items: List[Any]) -> List[bytes | str]:
        normalized: List[bytes | str] = []
        for item in items:
            if isinstance(item, str):
                normalized.append(item)
                continue
            if isinstance(item, dict):
                path = item.get("path")
                if isinstance(path, str) and os.path.exists(path):
                    normalized.append(path)
                    continue
                audio = item.get("array")
                sr = item.get("sampling_rate") or item.get("sample_rate") or 16000
                if isinstance(audio, np.ndarray):
                    wav_bytes = self._write_wav_bytes(audio, int(sr))
                    normalized.append(wav_bytes)
                    continue
            if isinstance(item, np.ndarray):
                wav_bytes = self._write_wav_bytes(item, 16000)
                normalized.append(wav_bytes)
                continue
            normalized.append("")
        return normalized

    def generate(self, input, **kwargs):
        if not isinstance(input, list):
            input = [input]

        items = self._normalize_inputs(input)
        results = []
        for obj in items:
            if isinstance(obj, str) and obj != "":
                with open(obj, "rb") as f:
                    resp = self.client.audio.transcriptions.create(
                        model=self.model,
                        file=f,
                    )
            elif isinstance(obj, (bytes, bytearray)) and len(obj) > 0:
                with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                    tmp.write(obj)
                    tmp.flush()
                    with open(tmp.name, "rb") as f:
                        resp = self.client.audio.transcriptions.create(
                            model=self.model,
                            file=f,
                        )
            else:
                resp = None

            results.append(
                {"text": getattr(resp, "text", "") if resp is not None else ""}
            )
        return results
