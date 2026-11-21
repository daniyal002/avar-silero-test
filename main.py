
import os
import torch

MODEL_URL = (
    "https://models.silero.ai/models/tts/"
    "cyr/v4_cyrillic.pt"
)

MODEL_FILE = "v4_cyrillic.pt"
SPEAKER = "b_ava"
SAMPLE_RATE = 48000

texts = [
    "Йорчӏами",
    "Йорчӏани",
    "Ворчӏами",
    "Ворчӏани",
    "Ча",
    "Чӏа",
    "Чи",
    "Чӏи",
    "Чӏам",
    "Чӏан",
    "Кин вугев мун?",
    "Дица жакъа кумек гьабуна дир эбелалъе телефон къачӏазе.",
]

torch.set_num_threads(4)
device = torch.device("cpu")

if not os.path.isfile(MODEL_FILE):
    print("Downloading Silero model...")
    torch.hub.download_url_to_file(
        MODEL_URL,
        MODEL_FILE
    )

print("Loading model...")

model = torch.package.PackageImporter(
    MODEL_FILE
).load_pickle("tts_models", "model")

model.to(device)

os.makedirs("output", exist_ok=True)

for i, text in enumerate(texts, start=1):
    print(f"{i:02d}: {text}")

    output_path = f"output/test_{i:02d}.wav"

    model.save_wav(
        text=text,
        speaker=SPEAKER,
        sample_rate=SAMPLE_RATE,
        audio_path=output_path,
    )

    print(f"Saved: {output_path}")

print("Done!")
