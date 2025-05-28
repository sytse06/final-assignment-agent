from smolagents import Tool
from openai import OpenAI
from .speech_recognition_tool import SpeechRecognitionTool
from io import BytesIO
import yt_dlp
import av
import torchaudio
import subprocess
import requests
import base64


class YoutubeVideoTool(Tool):
    name = "youtube_video"
    description = """Process the video and return the requested information from it."""
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL of the YouTube video.",
        },
        "query": {
            "type": "string",
            "description": "The question to answer.",
        },
    }
    output_type = "string"

    def __init__(
        self,
        video_quality: int = 360,
        frames_interval: int | float | None = 2,
        chunk_duration: int | float | None = 20,
        speech_recognition_tool: SpeechRecognitionTool | None = None,
        client: OpenAI | None = None,
        model_id: str = "gpt-4.1-mini",
        debug: bool = False,
        **kwargs,
    ):
        self.video_quality = video_quality
        self.speech_recognition_tool = speech_recognition_tool
        self.frames_interval = frames_interval
        self.chunk_duration = chunk_duration

        self.client = client or OpenAI()
        self.model_id = model_id

        self.debug = debug

        super().__init__(**kwargs)

    def forward(self, url: str, query: str):
        """
        Process the video and return the requested information.
        Args:
            url (str): The URL of the YouTube video.
            query (str): The question to answer.
        Returns:
            str: Answer to the query.
        """
        answer = ""
        for chunk in self._split_video_into_chunks(url):
            prompt = self._prompt(
                chunk,
                query,
                answer,
            )
            response = self.client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt,
                            },
                            *[
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{frame}",
                                }
                                for frame in self._base64_frames(chunk["frames"])
                            ],
                        ],
                    }
                ],
            )
            answer = response.output_text
            if self.debug:
                print(
                    f"CHUNK {chunk['start']} - {chunk['end']}:\n\n{prompt}\n\nANSWER:\n{answer}"
                )

        if answer.strip() == "I need to keep watching":
            answer = ""
        return answer

    def _prompt(self, chunk, query, aggregated_answer):
        prompt = [
            f"""\
These are some frames of a video that I want to upload.
I will ask a question about the entire video, but I will only last part of it.
Aggregate answer about the entire video, use information about previous parts but do not reference the previous parts in the answer directly.

Ground your answer based on video title, description, captions, vide frames or answer from previous parts.
If no evidences presented just say "I need to keep watching".

VIDEO TITLE:
{chunk["title"]}

VIDEO DESCRIPTION:
{chunk["description"]}

FRAMES SUBTITLES:
{chunk["captions"]}"""
        ]

        if aggregated_answer:
            prompt.append(f"""\
Here is the answer to the same question based on the previous video parts:
                          
BASED ON PREVIOUS PARTS:
{aggregated_answer}""")

        prompt.append(f"""\
                      
QUESTION:
{query}""")

        return "\n\n".join(prompt)

    def _split_video_into_chunks(
        self, url: str, with_captions: bool = True, with_frames: bool = True
    ):
        video = self._process_video(
            url, with_captions=with_captions, with_frames=with_frames
        )
        video_duration = video["duration"]
        chunk_duration = self.chunk_duration or video_duration

        chunk_start = 0.0
        while chunk_start < video_duration:
            chunk_end = min(chunk_start + chunk_duration, video_duration)
            chunk = self._get_video_chunk(video, chunk_start, chunk_end)
            yield chunk
            chunk_start += chunk_duration

    def _get_video_chunk(self, video, start, end):
        chunk_captions = [
            c for c in video["captions"] if c["start"] <= end and c["end"] >= start
        ]
        chunk_frames = [
            f
            for f in video["frames"]
            if f["timestamp"] >= start and f["timestamp"] <= end
        ]

        return {
            "title": video["title"],
            "description": video["description"],
            "start": start,
            "end": end,
            "captions": "\n".join([c["text"] for c in chunk_captions]),
            "frames": chunk_frames,
        }

    def _process_video(
        self, url: str, with_captions: bool = True, with_frames: bool = True
    ):
        lang = "en"
        info = self._get_video_info(url, lang)

        if with_captions:
            captions = self._extract_captions(
                lang, info.get("subtitles", {}), info.get("automatic_captions", {})
            )
            if not captions and self.speech_recognition_tool:
                audio_url = self._select_audio_format(info["formats"])
                audio = self._capture_audio(audio_url)
                waveform, sample_rate = torchaudio.load(audio)
                assert sample_rate == 16000
                waveform_np = waveform.squeeze().numpy()
                captions = self.speech_recognition_tool.transcribe(waveform_np)
        else:
            captions = []

        if with_frames:
            video_url = self._select_video_format(info["formats"], 360)["url"]
            frames = self._capture_video_frames(video_url, self.frames_interval)
        else:
            frames = []

        return {
            "id": info["id"],
            "title": info["title"],
            "description": info["description"],
            "duration": info["duration"],
            "captions": captions,
            "frames": frames,
        }

    def _get_video_info(self, url: str, lang: str):
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "format": "bestvideo[ext=mp4][height<=360]+bestaudio[ext=m4a]/best[height<=360]",
            "forceurl": True,
            "noplaylist": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitlesformat": "vtt",
            "subtitleslangs": [lang],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        return info

    def _extract_captions(self, lang, subtitles, auto_captions):
        caption_tracks = subtitles.get(lang) or auto_captions.get(lang) or []

        structured_captions = []

        srt_track = next(
            (track for track in caption_tracks if track["ext"] == "srt"), None
        )
        vtt_track = next(
            (track for track in caption_tracks if track["ext"] == "vtt"), None
        )

        if srt_track:
            import pysrt

            response = requests.get(srt_track["url"])
            response.raise_for_status()
            srt_data = response.content.decode("utf-8")

            def to_sec(t):
                return (
                    t.hours * 3600 + t.minutes * 60 + t.seconds + t.milliseconds / 1000
                )

            structured_captions = [
                {
                    "start": to_sec(sub.start),
                    "end": to_sec(sub.end),
                    "text": sub.text.strip(),
                }
                for sub in pysrt.from_str(srt_data)
            ]
        if vtt_track:
            import webvtt
            from io import StringIO

            response = requests.get(vtt_track["url"])
            response.raise_for_status()
            vtt_data = response.text

            vtt_file = StringIO(vtt_data)

            def to_sec(t):
                """Convert 'HH:MM:SS.mmm' to float seconds"""
                h, m, s = t.split(":")
                s, ms = s.split(".")
                return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

            for caption in webvtt.read_buffer(vtt_file):
                structured_captions.append(
                    {
                        "start": to_sec(caption.start),
                        "end": to_sec(caption.end),
                        "text": caption.text.strip(),
                    }
                )
        return structured_captions

    def _select_video_format(self, formats, video_quality):
        video_format = next(
            f
            for f in formats
            if f.get("vcodec") != "none" and f.get("height") == video_quality
        )
        return video_format

    def _capture_video_frames(self, video_url, capture_interval_sec=None):
        ffmpeg_cmd = [
            "ffmpeg",
            "-i",
            video_url,
            "-f",
            "matroska",  # container format
            "-",
        ]

        process = subprocess.Popen(
            ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

        container = av.open(process.stdout)
        stream = container.streams.video[0]
        time_base = stream.time_base

        frames = []
        next_capture_time = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue

            timestamp = float(frame.pts * time_base)
            if capture_interval_sec is None or timestamp >= next_capture_time:
                frames.append(
                    {
                        "timestamp": timestamp,
                        "image": frame.to_image(),  # PIL image
                    }
                )
                if capture_interval_sec is not None:
                    next_capture_time += capture_interval_sec

        process.terminate()
        return frames

    def _base64_frames(self, frames):
        base64_frames = []
        for f in frames:
            buffered = BytesIO()
            f["image"].save(buffered, format="JPEG")
            encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_frames.append(encoded)
        return base64_frames

    def _select_audio_format(self, formats):
        audio_formats = [
            f
            for f in formats
            if f.get("vcodec") == "none"
            and f.get("acodec")
            and f.get("acodec") != "none"
        ]

        if not audio_formats:
            raise ValueError("No valid audio-only formats found.")

        # Prefer m4a > webm, highest abr first
        preferred_exts = ["m4a", "webm"]

        def sort_key(f):
            ext_score = (
                preferred_exts.index(f["ext"]) if f["ext"] in preferred_exts else 99
            )
            abr = f.get("abr") or 0
            return (ext_score, -abr)

        audio_formats.sort(key=sort_key)
        return audio_formats[0]["url"]

    def _capture_audio(self, audio_url) -> BytesIO:
        audio_buffer = BytesIO()
        ffmpeg_audio_cmd = [
            "ffmpeg",
            "-i",
            audio_url,
            "-f",
            "wav",
            "-acodec",
            "pcm_s16le",  # Whisper prefers PCM
            "-ac",
            "1",  # Mono
            "-ar",
            "16000",  # 16kHz for Whisper
            "-",
        ]

        result = subprocess.run(
            ffmpeg_audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            raise RuntimeError("ffmpeg failed:\n" + result.stderr.decode())

        audio_buffer = BytesIO(result.stdout)
        audio_buffer.seek(0)
        return audio_buffer
