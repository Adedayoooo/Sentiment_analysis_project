from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

MODEL_PATH = "results"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

review = ""
inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits
probs = torch.softmax(logits, dim=1)
predicted_class_id = torch.argmax(probs, dim=1).item()
confidence = (probs[0][predicted_class_id].item())*100
sentiment_map = {0: "NEGATIVE", 1: "POSITIVE"}
print(f"User review:\n'{review}'.\nThis is a {sentiment_map[predicted_class_id]} movie review, predicted with confidence level of {confidence:.2f}%")
