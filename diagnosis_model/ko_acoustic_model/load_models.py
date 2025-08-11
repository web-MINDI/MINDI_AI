import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from ko_acoustic_model.lgbm_model import VoiceClassifier
from ko_acoustic_model.extract_feature import extract_features


def load_models(vit_path, lgb_path, example_wav_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vit_model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    vit_model.load_state_dict(torch.load(vit_path, map_location=device))
    vit_model.to(device)
    vit_model.eval()

    feats = extract_features(example_wav_path)
    input_dim = feats.shape[1]

    lgbm_model = VoiceClassifier(input_dim = input_dim)
    lgbm_model.load_state_dict(torch.load(lgb_path, map_location=device))
    lgbm_model.to(device)
    lgbm_model.eval()

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    return vit_model, lgbm_model, device, image_transform

