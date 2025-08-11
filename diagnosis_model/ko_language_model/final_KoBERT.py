import os
import json
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
import torch
import json



# 예측 함수
def KoBERT_final(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
    model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=2)
    model.load_state_dict(torch.load("diagnosis_model/ko_language_model/final_KoBERT.pt", map_location=device))
    model.to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=200)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()
    return 2 if pred==1 else 0

