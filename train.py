import json
import logging
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import shutil
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s--%(levelname)s--%(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_and_preprocess():
    logger.info("Loading dataset...")
    df = pd.read_csv("/kaggle/input/datasets/adedayoadebayo23/imdb-dataset/IMDB Dataset.csv")
    label_map = {"negative": 0, "positive": 1}
    df["sentiment"] = df["sentiment"].map(label_map)
    texts = df["review"].tolist()
    labels = df["sentiment"].tolist()
    return texts, labels

def prepare_split(texts, labels):
    return train_test_split(
        texts, labels,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=labels
    )

def tokenize_data(X_train, X_test):
    tokenizer = BertTokenizer.from_pretrained(config["model_name"])
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=config["max_length"])
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=config["max_length"])
    return train_encodings, test_encodings, tokenizer

def build_dataset(encodings, labels):
    dataset = []
    for i in range(len(labels)):
        item = {key: torch.tensor(val[i]) for key, val in encodings.items()}
        item["labels"] = torch.tensor(labels[i])
        dataset.append(item)
    return dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def train_model():
    texts, labels = load_and_preprocess()
    X_train, X_test, y_train, y_test = prepare_split(texts, labels)
    train_encodings, test_encodings, tokenizer = tokenize_data(X_train, X_test)
    train_dataset = build_dataset(train_encodings, y_train)
    test_dataset = build_dataset(test_encodings, y_test)

    model = BertForSequenceClassification.from_pretrained(config["model_name"], num_labels=2).to(device)

    training_args = TrainingArguments(
        output_dir="./final_model",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        logging_steps=100,
        warmup_ratio=0.1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Saving model and tokenizer...")
    trainer.save_model("/kaggle/working/final_model")
    tokenizer.save_pretrained("/kaggle/working/final_model")
    logger.info("Training completed.")

if __name__ == "__main__":
    train_model()

