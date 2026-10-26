import os
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset

# --- CONFIGURATION ---
MODEL_NAME = "answerdotai/ModernBERT-large"
REAL_DATA = "real_10k.jsonl"
SYNTH_DATA = "synth_10k.jsonl"
OUTPUT_DIR = "MODERNBERT_MITRE_COMMUNITY"

# Paramètres suggérés (à mettre à jour après optuna_tune_20k.py)
PARAMS = {
    'learning_rate': 3.5e-05,
    'weight_decay': 0.05,
    'num_train_epochs': 5,
    'batch_size': 16
}

def load_data():
    all_data = []
    for f in [REAL_DATA, SYNTH_DATA]:
        with open(f, 'r', encoding='utf-8') as fin:
            for line in fin:
                all_data.append(json.loads(line))
    
    all_labels = sorted(list(set(l for item in all_data for l in item['labels'])))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {i: l for i, l in enumerate(all_labels)}
    
    texts = [x['text'] for x in all_data]
    matrix = np.zeros((len(texts), len(all_labels)), dtype=np.float32)
    for i, item in enumerate(all_data):
        for lbl in item['labels']:
            matrix[i, label2id[lbl]] = 1.0
            
    return texts, matrix, id2label, label2id

def train():
    texts, labels, id2label, label2id = load_data()
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = Dataset.from_dict({"text": train_texts, "labels": train_labels}).map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=1024), batched=True
    )
    val_ds = Dataset.from_dict({"text": val_texts, "labels": val_labels}).map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=1024), batched=True
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(id2label), problem_type="multi_label_classification",
        id2label=id2label, label2id=label2id
    )
    
    args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}_checkpoints",
        per_device_train_batch_size=PARAMS['batch_size'],
        learning_rate=PARAMS['learning_rate'],
        weight_decay=PARAMS['weight_decay'],
        num_train_epochs=PARAMS['num_train_epochs'],
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        report_to="none"
    )
    
    def compute_metrics(eval_pred):
        preds = (1 / (1 + np.exp(-eval_pred.predictions)) > 0.5).astype(int)
        return {'f1_micro': f1_score(eval_pred.label_ids, preds, average='micro', zero_division=0)}
    
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )
    
    print("🚀 Démarrage de l'entraînement communautaire...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Modèle sauvegardé dans {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
