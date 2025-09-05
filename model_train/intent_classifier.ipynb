# train_intent_classifier.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    get_scheduler,
    Trainer,
    TrainingArguments,
)

from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

# =========================
# 1. Config
# =========================
MODEL_NAME = "xlm-roberta-base"   # or "xlm-roberta-large" if strong GPU
MAX_LEN = 128
BATCH_SIZE = 16
LR = 2e-5
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. Dataset Class
# =========================
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# =========================
# 3. Load Data
# =========================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

num_labels = train_df["intent_label"].nunique()

tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

train_dataset = IntentDataset(train_df["QueryText"].tolist(), train_df["intent_label"].tolist(), tokenizer, MAX_LEN)
test_dataset = IntentDataset(test_df["QueryText"].tolist(), test_df["intent_label"].tolist(), tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# =========================
# 4. Model Setup
# =========================
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)
num_training_steps = EPOCHS * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# =========================
# 5. Training Loop
# =========================
for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(train_loader, leave=True)
    total_loss = 0

    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Average Training Loss: {avg_loss:.4f}")

    # =========================
    # 6. Evaluation
    # =========================
    model.eval()
    preds, true_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average="macro")
    print(f"Validation Accuracy: {acc:.4f}, F1 (macro): {f1:.4f}")
    print(classification_report(true_labels, preds))

# =========================
# 7. Save Model
# =========================
model.save_pretrained("xlm_intent_model")
tokenizer.save_pretrained("xlm_intent_model")
print("Model saved in xlm_intent_model/")
