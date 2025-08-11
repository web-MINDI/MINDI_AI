import numpy as np
import torch
from ko_acoustic_model.remove_silence import remove_silence
from ko_acoustic_model.make_spectrogram import make_spectrogram_chunks
from ko_acoustic_model.extract_feature import extract_features

def predict_dementia(audio_path, vit_model, lgbm_model, device, image_transform):
    samples, sr = remove_silence(audio_path)
    images = make_spectrogram_chunks(samples, sr, image_transform)
    if not images:
        print("No valid spectrogram chunks for ViT.")
        return None
    vit_inputs = torch.stack(images).to(device)
    with torch.no_grad():
        outputs = vit_model(vit_inputs)
        probs_vit = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
    vit_final = np.mean(probs_vit)

    feats = extract_features(audio_path)
    if feats.size == 0:
        print("No valid acoustic segments for LightGBM.")
        return None
    feats_tensor = torch.tensor(feats, dtype = torch.float32).to(device)
    with torch.no_grad():
        outputs = lgbm_model(feats_tensor).squeeze().cpu().numpy()
    lgb_final = np.mean(outputs)

    final_prob = 0.5 * lgb_final + 0.5 * vit_final

    return round(final_prob, 4), lgb_final, vit_final
