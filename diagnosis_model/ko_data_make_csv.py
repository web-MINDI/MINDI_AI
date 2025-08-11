import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# 메인 모델에 사용될 wav to text 코드
def transcribe_audio(file_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained("openai/whisper-base", force_download=False)
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base", force_download=False).to(device)
    try:
        import librosa
        waveform, sr = librosa.load(file_path, sr=16000)
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        # Force English + transcribe
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
        outputs = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        return decoded

    except Exception as e:
        print(f"[Error] {file_path}: {e}")
        return ""

def single_wav_to_text(file_path):
    transcription = transcribe_audio(file_path)
    # print(transcription)
    return transcription
