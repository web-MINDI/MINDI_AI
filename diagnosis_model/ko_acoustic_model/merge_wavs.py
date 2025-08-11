from pydub import AudioSegment

def merge_wavs(audio_paths, output_path="merged.wav"):
    combined = AudioSegment.empty()
    for path in audio_paths:
        sound = AudioSegment.from_wav(path)
        combined += sound
    combined.export(output_path, format="wav")
    return output_path