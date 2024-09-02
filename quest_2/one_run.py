import whisper_timestamped as whisper
import librosa
import numpy as np
import soundfile as sf
import json

# Load the audio file
audio, sr = librosa.load("audio_recorded.wav", sr=None)

# Perform noise reduction
# This is a simple method; more advanced techniques exist
def reduce_noise(y, sr):
    # Compute the spectrogram of the audio
    D = librosa.stft(y)
    # Estimate the noise floor
    noise_floor = np.mean(np.abs(D[:, :10]), axis=1, keepdims=True)
    # Perform noise gating
    D_denoised = D * (np.abs(D) > noise_floor)
    # Reconstruct the audio from the denoised spectrogram
    y_denoised = librosa.istft(D_denoised)
    return y_denoised

# Apply noise reduction
audio_denoised = reduce_noise(audio, sr)

# Save the denoised audio
sf.write("audio_denoised.wav", audio_denoised, sr)

# Load the denoised audio for Whisper
audio = whisper.load_audio("audio_denoised.wav")

model = whisper.load_model("base", device="cpu")

result = whisper.transcribe(
    model, audio, language="vi", best_of=5, beam_size=5, temperature=0.1, condition_on_previous_text=False, compression_ratio_threshold=0.,
)

print(json.dumps(result, indent=2, ensure_ascii=False))