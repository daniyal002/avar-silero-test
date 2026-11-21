
import json
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg
import noisereduce as nr
import numpy as np
import soundfile as sf
import torch


# ==========================================
# CONFIG
# ==========================================

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

VIDEO = Path("input.mp4")
SEGMENTS = Path("segments.json")
MODEL_FILE = Path("v4_cyrillic.pt")

OUT = Path("output")
AUDIO = Path("audio")

SR = 48000
SPEAKER = "b_ava"

# Длительность тестового видео.
# Установи None для обработки всего видео.
MAX_VIDEO_DURATION = 45

# Настройки очистки звука
NOISE_REDUCTION = 0.65

# Плавное появление и затухание звука
FADE_MS = 15

OUT.mkdir(parents=True, exist_ok=True)
AUDIO.mkdir(parents=True, exist_ok=True)


# ==========================================
# HELPERS
# ==========================================

def run(*args):
    command = [str(arg) for arg in args]
    subprocess.run(command, check=True)


def get_video_duration(path):
    result = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
    )

    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
        result.stderr,
    )

    if not match:
        raise RuntimeError(
            "Не удалось определить длительность видео"
        )

    hours, minutes, seconds = match.groups()

    return (
        int(hours) * 3600
        + int(minutes) * 60
        + float(seconds)
    )


def apply_fade(audio, sample_rate, fade_ms=15):
    audio = audio.copy()

    fade_samples = min(
        int(sample_rate * fade_ms / 1000),
        len(audio) // 2,
    )

    if fade_samples > 0:
        fade_in = np.linspace(
            0.0, 1.0, fade_samples
        )

        fade_out = np.linspace(
            1.0, 0.0, fade_samples
        )

        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out

    return audio


def calculate_timing(items):
    """
    Анализ длительности реплик и пересечений.
    Результаты предназначены для будущего UI.
    """

    generated = [
        item for item in items
        if item.get("status") == "generated"
    ]

    generated.sort(
        key=lambda item: float(item["start"])
    )

    for item in generated:
        item["overlap_next_seconds"] = 0.0
        item["overlap_next_id"] = None

    for i, item in enumerate(generated[:-1]):
        next_item = generated[i + 1]

        actual_end = (
            float(item["start"])
            + float(item["tts_duration"])
        )

        overlap = max(
            0.0,
            min(
                actual_end,
                float(next_item["start"])
                + float(next_item["tts_duration"]),
            ) - float(next_item["start"])
        )

        item["overlap_next_seconds"] = round(
            overlap, 3
        )

        if overlap > 0:
            item["overlap_next_id"] = next_item["id"]

            print(
                f"WARNING: segments "
                f"{item['id']} and {next_item['id']} "
                f"overlap by {overlap:.2f}s"
            )


# ==========================================
# VALIDATION
# ==========================================

if not VIDEO.exists():
    raise FileNotFoundError(VIDEO)

if not SEGMENTS.exists():
    raise FileNotFoundError(SEGMENTS)

if not MODEL_FILE.exists():
    raise FileNotFoundError(MODEL_FILE)

items = json.loads(
    SEGMENTS.read_text(encoding="utf-8")
)

video_duration = get_video_duration(VIDEO)

if MAX_VIDEO_DURATION is not None:
    output_duration = min(
        video_duration,
        MAX_VIDEO_DURATION,
    )
else:
    output_duration = video_duration

print(f"Video duration: {video_duration:.2f}s")
print(f"Output duration: {output_duration:.2f}s")


# ==========================================
# LOAD SILERO
# ==========================================

print("Loading Silero...")

torch.set_num_threads(4)

model = torch.package.PackageImporter(
    str(MODEL_FILE)
).load_pickle(
    "tts_models",
    "model",
)

model.to(torch.device("cpu"))


# ==========================================
# AUDIO TRACK
# ==========================================

track = np.zeros(
    int(np.ceil(output_duration * SR)),
    dtype=np.float32,
)


# ==========================================
# GENERATE SEGMENTS
# ==========================================

for item in items:
    segment_id = item["id"]
    text = item.get("ava", "").strip()

    start = float(item["start"])
    end = float(item["end"])

    # Значения для будущего интерфейса
    item["status"] = "pending"
    item["tts_duration"] = None
    item["available_duration"] = round(
        end - start, 3
    )
    item["overflow_seconds"] = 0.0
    item["overflow_percent"] = 0.0
    item["overlap_next_seconds"] = 0.0
    item["overlap_next_id"] = None
    item["actual_end"] = None
    item["truncated_by_video"] = False

    if not text:
        item["status"] = "untranslated"
        continue

    if start < 0 or end <= start or end > video_duration:
        raise ValueError(
            f"Invalid timestamps: {item}"
        )

    if start >= output_duration:
        item["status"] = "outside_output"
        continue

    raw_path = AUDIO / f"{segment_id:03d}_raw.wav"
    clean_path = AUDIO / f"{segment_id:03d}_clean.wav"

    print(f"\nGenerating {segment_id}: {text}")

    # ======================================
    # SILERO TTS
    # ======================================

    model.save_wav(
        text=text,
        speaker=SPEAKER,
        sample_rate=SR,
        audio_path=str(raw_path),
    )

    audio, rate = sf.read(raw_path)

    if rate != SR:
        raise ValueError(
            f"Unexpected sample rate: {rate}"
        )

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # ======================================
    # NOISE REDUCTION
    # ======================================

    clean = nr.reduce_noise(
        y=audio,
        sr=SR,
        stationary=False,
        prop_decrease=NOISE_REDUCTION,
    ).astype(np.float32)

    sf.write(clean_path, clean, SR)

    # ======================================
    # TIMING ANALYSIS
    # ======================================

    available = end - start
    actual = len(clean) / SR

    overflow = max(
        0.0,
        actual - available,
    )

    overflow_percent = (
        overflow / available * 100
        if available > 0
        else 0.0
    )

    item["tts_duration"] = round(actual, 3)
    item["available_duration"] = round(
        available, 3
    )
    item["overflow_seconds"] = round(
        overflow, 3
    )
    item["overflow_percent"] = round(
        overflow_percent, 1
    )
    item["actual_end"] = round(
        start + actual, 3
    )

    if overflow > 0:
        print(
            f"WARNING: segment {segment_id} "
            f"exceeds timing by "
            f"{overflow:.2f}s "
            f"({overflow_percent:.1f}%). "
            f"Available: {available:.2f}s, "
            f"generated: {actual:.2f}s."
        )
    else:
        print(
            f"OK: segment {segment_id} "
            f"fits timing "
            f"({actual:.2f}s / {available:.2f}s)"
        )

    # ======================================
    # PLACE AUDIO ON TIMELINE
    # ======================================

    # Не ускоряем и не обрезаем
    # реплику по её исходному интервалу.
    clean = apply_fade(
        clean,
        SR,
        FADE_MS,
    )

    start_sample = round(start * SR)
    end_sample = start_sample + len(clean)

    # Ограничиваем только концом видео.
    stop = min(
        end_sample,
        len(track),
    )

    length = max(
        0,
        stop - start_sample,
    )

    if length > 0:
        track[start_sample:stop] += clean[:length]

    if length < len(clean):
        item["truncated_by_video"] = True

        print(
            f"WARNING: segment {segment_id} "
            "is truncated by video end."
        )

    item["status"] = "generated"


# ==========================================
# CHECK OVERLAPS
# ==========================================

calculate_timing(items)


# ==========================================
# SAVE TIMING REPORT
# ==========================================

timing_path = OUT / "segments_timing.json"

timing_path.write_text(
    json.dumps(
        items,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(f"\nTiming report: {timing_path}")


# ==========================================
# NORMALIZE AUDIO
# ==========================================

if track.size:
    peak = float(np.max(np.abs(track)))

    if peak > 0.95:
        track *= 0.95 / peak


# ==========================================
# SAVE WAV
# ==========================================

dub_path = OUT / "avar_dub.wav"

sf.write(
    dub_path,
    track,
    SR,
)

print(f"Audio saved: {dub_path}")


# ==========================================
# BUILD VIDEO
# ==========================================

output_video = OUT / "dubbed.mp4"

run(
    FFMPEG,
    "-y",
    "-hide_banner",
    "-loglevel", "error",
    "-i", VIDEO,
    "-i", dub_path,
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-c:v", "copy",
    "-c:a", "aac",
    "-b:a", "192k",
    "-t", str(output_duration),
    output_video,
)

print(f"\nDone: {output_video}")
