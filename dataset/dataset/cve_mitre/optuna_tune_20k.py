import os
import json
import torch
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset

# --- CONFIGURATION ---
MODEL_NAME = "answerdotai/ModernBERT-large"
REAL_DATA = "real_10k.jsonl"
SYNTH_DATA = "synth_10k.jsonl"
N_TRIALS = 15

def load_data():
    print("📂 Chargement du pack 20k (10k Réel + 10k Synthétique)...")
    all_data = []
    for f in [REAL_DATA, SYNTH_DATA]:
        with open(f, 'r', encoding='utf-8') as fin:
            for line in fin:
                all_data.append(json.loads(line))
    
    # Extraction des labels uniques
    all_labels = set()
    for item in all_data:
        all_labels.update(item['labels'])
    
    sorted_labels = sorted(list(all_labels))
    label2id = {l: i for i, l in enumerate(sorted_labels)}
    id2label = {i: l for i, l in enumerate(sorted_labels)}
    
    texts = [x['text'] for x in all_data]
    labels_matrix = np.zeros((len(texts), len(sorted_labels)), dtype=np.float32)
    for i, item in enumerate(all_data):
        for lbl in item['labels']:
            labels_matrix[i, label2id[lbl]] = 1.0
            
    print(f"📊 Dataset prêt : 20,000 exemples, {len(sorted_labels)} classes.")
    return texts, labels_matrix, id2label, label2id

def objective(trial, train_ds, val_ds, id2label, label2id):
    lr = trial.suggest_float('lr', 1e-5, 5e-5, log=True)
    wd = trial.suggest_float('weight_decay', 0.01, 0.1)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(id2label), problem_type="multi_label_classification",
        id2label=id2label, label2id=label2id
    )
    
    args = TrainingArguments(
        output_dir=f"tuning/trial_{trial.number}",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=lr,
        weight_decay=wd,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none"
    )
    
    def compute_metrics(eval_pred):
        preds = (1 / (1 + np.exp(-eval_pred.predictions)) > 0.5).astype(int)
        return {'f1_micro': f1_score(eval_pred.label_ids, preds, average='micro', zero_division=0)}

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    return trainer.evaluate()['eval_f1_micro']

def main():
    texts, labels, id2label, label2id = load_data()
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.15, random_state=42)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = Dataset.from_dict({"text": train_texts, "labels": train_labels}).map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=512), batched=True
    )
    val_ds = Dataset.from_dict({"text": val_texts, "labels": val_labels}).map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=512), batched=True
    )

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda t: objective(t, train_ds, val_ds, id2label, label2id), n_trials=N_TRIALS)
    
    print("\n🏆 MEILLEURS PARAMÈTRES TROUVÉS :")
    print(study.best_params)

if __name__ == "__main__":
    main()
