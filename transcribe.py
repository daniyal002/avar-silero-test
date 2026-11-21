
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from faster_whisper import WhisperModel

VIDEO = Path("input.mp4")
OUTPUT = Path("segments.json")

def extract_audio(video_path: Path) -> np.ndarray:
    """Извлекает аудио через FFmpeg в mono PCM 16 kHz."""

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg,
        "-v", "error",
        "-i", str(video_path),
        "-vn",
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", "16000",
        "pipe:1",
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    audio = np.frombuffer(
        result.stdout,
        dtype=np.float32,
    ).copy()

    if audio.size == 0:
        raise RuntimeError("FFmpeg не извлёк аудио из видео")

    return audio


def main():
    if not VIDEO.exists():
        raise FileNotFoundError(VIDEO)

    print("Extracting audio...")
    audio = extract_audio(VIDEO)

    print(f"Audio duration: {len(audio) / 16000:.2f}s")

    print("Loading Whisper...")
    model = WhisperModel(
        "medium",
        device="cpu",
        compute_type="int8",
    )

    print("Transcribing...")

    segments, info = model.transcribe(
        audio,
        language="ru",
        vad_filter=False,
        beam_size=5,
        condition_on_previous_text=False,
        word_timestamps=True,
        hallucination_silence_threshold=2.0,
    )

    result = []

    for index, segment in enumerate(segments, start=1):
        item = {
            "id": index,
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "ru": segment.text.strip(),
            "ava": "",
        }

        result.append(item)

        print(
            f"[{item['start']:.2f} - {item['end']:.2f}] "
            f"{item['ru']}"
        )

    OUTPUT.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved {len(result)} segments to {OUTPUT}")


if __name__ == "__main__":
    main()
