import os

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

def transcribe_audio(file_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)

    waveform, sr = librosa.load(file_path, sr=16000)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    outputs = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

    decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    return decoded

def single_wav_to_text(file_path):
    transcription = transcribe_audio(file_path)
    print(transcription)
    return transcription