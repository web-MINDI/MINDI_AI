import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def remove_silence(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    nonsilent = detect_nonsilent(audio, min_silence_len=300, silence_thresh=audio.dBFS - 14)
    voiced_audio = AudioSegment.empty()
    for start, end in nonsilent:
        voiced_audio += audio[start:end]
    samples = np.array(voiced_audio.get_array_of_samples()).astype(np.float32)
    samples /= np.max(np.abs(samples))
    sr = voiced_audio.frame_rate
    return samples, sr