import logging
import os
import asyncio
import tempfile
import soundfile as sf
import numpy as np
from typing import List, Any
from pathlib import Path
from dataclasses import dataclass

try:
    import edge_tts
except ImportError:
    raise ImportError("edge-tts not installed. Run: pip install edge-tts")

from src.podcast.script_generator import PodcastScript

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    speaker: str
    text: str
    audio_data: Any
    duration: float
    file_path: str


class PodcastTTSGenerator:
    def __init__(self, lang_code: str = "a", sample_rate: int = 24000):
        # lang_code kept for signature compat — edge-tts handles language via voice name
        self.sample_rate = sample_rate

        self.speaker_voices = {
            "Speaker 1": "en-US-JennyNeural",   # Female
            "Speaker 2": "en-US-GuyNeural"       # Male
        }

        logger.info("PodcastTTSGenerator initialized with edge-tts (free, no API key)")

    def generate_podcast_audio(
        self,
        podcast_script: PodcastScript,
        output_dir: str = "outputs/podcast_audio",
        combine_audio: bool = True
    ) -> List[str]:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating podcast audio for {podcast_script.total_lines} segments")

        audio_segments = []
        output_files = []

        for i, line_dict in enumerate(podcast_script.script):
            speaker, dialogue = next(iter(line_dict.items()))
            logger.info(f"Processing segment {i+1}/{podcast_script.total_lines}: {speaker}")

            try:
                segment_path = os.path.join(
                    output_dir,
                    f"segment_{i+1:03d}_{speaker.replace(' ', '_').lower()}.wav"
                )
                audio_data = self._generate_single_segment(speaker, dialogue, segment_path)
                output_files.append(segment_path)

                if combine_audio:
                    audio_segments.append(AudioSegment(
                        speaker=speaker,
                        text=dialogue,
                        audio_data=audio_data,
                        duration=len(audio_data) / self.sample_rate,
                        file_path=segment_path
                    ))

                logger.info(f"Generated segment {i+1}: {os.path.basename(segment_path)}")

            except Exception as e:
                logger.error(f"Failed to generate segment {i+1}: {str(e)}")
                continue

        if combine_audio and audio_segments:
            combined_path = self._combine_audio_segments(audio_segments, output_dir)
            output_files.append(combined_path)

        logger.info(f"Podcast generation complete — {len(output_files)} files")
        return output_files

    def _generate_single_segment(self, speaker: str, text: str, output_path: str) -> np.ndarray:
        voice = self.speaker_voices.get(speaker, "en-US-JennyNeural")
        clean_text = self._clean_text_for_tts(text)

        # edge-tts writes mp3 to a temp file; we convert to wav numpy array
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        async def _synthesize():
            communicate = edge_tts.Communicate(clean_text, voice)
            await communicate.save(tmp_path)

        asyncio.run(_synthesize())

        # Read mp3 -> wav via soundfile (requires libsndfile with mp3 support)
        # Fallback: write mp3 directly if wav conversion fails
        try:
            data, sr = sf.read(tmp_path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float32)
            sf.write(output_path, data, sr)
            os.unlink(tmp_path)
            return data
        except Exception:
            # soundfile can't read mp3 — save as mp3 instead
            mp3_path = output_path.replace(".wav", ".mp3")
            os.rename(tmp_path, mp3_path)
            logger.warning(f"Saved as mp3 (install ffmpeg for wav): {mp3_path}")
            # Return silence placeholder for combine step
            return np.zeros(int(0.5 * self.sample_rate), dtype=np.float32)

    def _clean_text_for_tts(self, text: str) -> str:
        clean = text.strip()
        clean = clean.replace("...", ".").replace("!!", "!").replace("??", "?")
        if not clean.endswith((".", "!", "?")):
            clean += "."
        return clean

    def _combine_audio_segments(self, segments: List[AudioSegment], output_dir: str) -> str:
        logger.info(f"Combining {len(segments)} audio segments")
        pause = np.zeros(int(0.3 * self.sample_rate), dtype=np.float32)

        parts = []
        for i, seg in enumerate(segments):
            parts.append(seg.audio_data)
            if i < len(segments) - 1:
                parts.append(pause)

        final = np.concatenate(parts)
        combined_path = os.path.join(output_dir, "complete_podcast.wav")
        sf.write(combined_path, final, self.sample_rate)
        duration = len(final) / self.sample_rate
        logger.info(f"Combined podcast saved: {combined_path} ({duration:.1f}s)")
        return combined_path