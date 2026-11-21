import soundfile as sf
import noisereduce as nr

audio, sr = sf.read("output/test_12.wav")

clean = nr.reduce_noise(
    y=audio,
    sr=sr,
    stationary=False,
    prop_decrease=0.65,
)

sf.write("output/test_12_clean.wav", clean, sr)