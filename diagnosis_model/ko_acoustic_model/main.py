from ko_acoustic_model.load_models import *
from ko_acoustic_model.predict_dementia import*
from ko_acoustic_model.get_score import *
from ko_acoustic_model.merge_wavs import *

def ko_acoustic_model(audio_paths):
    vit_path = "diagnosis_model/ko_acoustic_model/vit_model_ver2.pth"
    lgb_path = "diagnosis_model/ko_acoustic_model/lgbm_model_ver2.pt"

    merged_path = merge_wavs(audio_paths, output_path="merged.wav")
    vit_model, lgbm_model, device, image_transform = load_models(vit_path, lgb_path,  example_wav_path=merged_path)

    # probs = []
    # for audio_path in audio_paths:
    #     prob = predict_dementia(merged_path, vit_model, lgbm_model, device, image_transform)
    #     if prob is not None:
    #         probs.append(prob)
    prob, lgb_final, vit_final = predict_dementia(merged_path, vit_model, lgbm_model, device, image_transform)

    if prob is not None:
        final_prob = round(prob, 4)
        score = get_score(final_prob)
        # print(f"Dementia Probability: {final_prob}")
        # print(f"Score: {score}")

    return score, lgb_final, vit_final

