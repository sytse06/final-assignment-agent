from smolagents import Tool
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, logging
import warnings


class SpeechRecognitionTool(Tool):
    name = "speech_to_text"
    description = """Transcribes speech from audio."""

    inputs = {
        "audio": {
            "type": "string",
            "description": "Path to the audio file to transcribe.",
        },
        "with_time_markers": {
            "type": "boolean",
            "description": "Whether to include timestamps in the transcription output. Each timestamp appears on its own line in the format [float, float], indicating the number of seconds elapsed from the start of the audio.",
            "nullable": True,
            "default": False,
        },
    }
    output_type = "string"

    chunk_length_s = 30

    def __new__(cls, *args, **kwargs):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3-turbo"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id)

        logging.set_verbosity_error()
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r".*The input name `inputs` is deprecated.*",
        )
        cls.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=cls.chunk_length_s,
            return_timestamps=True,
        )

        return super().__new__(cls, *args, **kwargs)

    def forward(self, audio: str, with_time_markers: bool = False) -> str:
        """
        Transcribes speech from audio.

        Args:
            audio (str): Path to the audio file to transcribe.
            with_time_markers (bool): Whether to include timestamps in the transcription output. Each timestamp appears on its own line in the format [float], indicating the number of seconds elapsed from the start of the audio.

        Returns:
            str: The transcribed text.
        """
        result = self.pipe(audio)
        if not with_time_markers:
            return result["text"].strip()

        txt = ""
        for chunk in self._normalize_chunks(result["chunks"]):
            txt += f"[{chunk['start']:.2f}]\n{chunk['text']}\n[{chunk['end']:.2f}]\n"
        return txt.strip()

    def transcribe(self, audio, **kwargs):
        result = self.pipe(audio, **kwargs)
        return self._normalize_chunks(result["chunks"])

    def _normalize_chunks(self, chunks):
        chunk_length_s = self.chunk_length_s
        absolute_offset = 0.0
        chunk_offset = 0.0
        normalized = []

        for chunk in chunks:
            timestamp_start = chunk["timestamp"][0]
            timestamp_end = chunk["timestamp"][1]
            if timestamp_start < chunk_offset:
                absolute_offset += chunk_length_s
                chunk_offset = timestamp_start
            absolute_start = absolute_offset + timestamp_start

            if timestamp_end < timestamp_start:
                absolute_offset += chunk_length_s
            absolute_end = absolute_offset + timestamp_end
            chunk_offset = timestamp_end

            chunk_text = chunk["text"].strip()
            if chunk_text:
                normalized.append(
                    {
                        "start": absolute_start,
                        "end": absolute_end,
                        "text": chunk_text,
                    }
                )

        return normalized
