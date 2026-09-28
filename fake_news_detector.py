# Fake News Detection — NLP and Deep Learning
# Ilayda Esma Asaroglu (2111742) · Liluz Atan (2186522)
# AI Lab, Sapienza University of Rome, 2025/2026

import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ── Config ────────────────────────────────────────────────────────────────────

SEED       = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 512
BATCH_SIZE = 16
EPOCHS     = 3
LR         = 2e-5

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_isot(true_path="True.csv", fake_path="Fake.csv"):
    real = pd.read_csv(true_path)
    real["label"] = 0
    fake = pd.read_csv(fake_path)
    fake["label"] = 1
    df = pd.concat([real, fake], ignore_index=True)
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df = df[["text", "label"]].dropna()
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"ISOT: {len(df)} articles — Real: {(df.label==0).sum()}, Fake: {(df.label==1).sum()}")
    return df


def load_welfake(path="WELFake_Dataset.csv"):
    df = pd.read_csv(path)
    df["label"] = 1 - df["label"]          # WELFake: 0=Fake,1=Real → flip to match ISOT
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df = df[["text", "label"]].dropna()
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"WELFake: {len(df)} articles — Real: {(df.label==0).sum()}, Fake: {(df.label==1).sum()}")
    return df


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df):
    df = df.copy()
    df["text"] = df["text"].apply(clean_text)
    return df[df["text"].str.len() > 20].reset_index(drop=True)


def split_data(df):
    X, y = df["text"].tolist(), df["label"].tolist()
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.10, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.10, random_state=SEED, stratify=y_tmp
    )
    print(f"Train: {len(X_train)} · Val: {len(X_val)} · Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Baseline ──────────────────────────────────────────────────────────────────

def train_baseline(X_train, y_train, X_val, y_val):
    print("\n[Baseline] TF-IDF + Logistic Regression")
    vec = TfidfVectorizer(
        max_features=50000, ngram_range=(1, 2),
        min_df=2, max_df=0.95, sublinear_tf=True
    )
    X_tr = vec.fit_transform(X_train)
    X_v  = vec.transform(X_val)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                             random_state=SEED, n_jobs=-1)
    clf.fit(X_tr, y_train)

    preds = clf.predict(X_v)
    print(f"  Val — Acc: {accuracy_score(y_val, preds):.4f}  "
          f"F1: {f1_score(y_val, preds, average='weighted'):.4f}")
    return vec, clf


def eval_baseline(vec, clf, X, y, label="Test"):
    preds = clf.predict(vec.transform(X))
    _print_metrics(y, preds, f"Baseline [{label}]")
    _confusion_matrix(y, preds, f"Baseline — {label}")
    return _metrics(y, preds)


# ── DistilBERT ────────────────────────────────────────────────────────────────

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts, self.labels, self.tok = texts, labels, tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx], max_length=MAX_LEN,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _compute_metrics(ep):
    preds = np.argmax(ep.predictions, axis=1)
    return {
        "accuracy":  accuracy_score(ep.label_ids, preds),
        "f1":        f1_score(ep.label_ids, preds, average="weighted"),
        "precision": precision_score(ep.label_ids, preds, average="weighted"),
        "recall":    recall_score(ep.label_ids, preds, average="weighted"),
    }


def train_distilbert(X_train, y_train, X_val, y_val):
    print("\n[DistilBERT] Fine-tuning")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    tok   = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    # "eager" attention is needed so the model can return attention weights
    # for the interpretability analysis below.
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, attn_implementation="eager"
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./output",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LR,
            warmup_ratio=0.1,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=100,
            seed=SEED,
            report_to="none",
        ),
        train_dataset=NewsDataset(X_train, y_train, tok),
        eval_dataset=NewsDataset(X_val,   y_val,   tok),
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    return trainer, tok


def eval_distilbert(trainer, tok, X, y, label="Test"):
    out   = trainer.predict(NewsDataset(X, y, tok))
    preds = np.argmax(out.predictions, axis=1)
    _print_metrics(y, preds, f"DistilBERT [{label}]")
    _confusion_matrix(y, preds, f"DistilBERT — {label}")
    return _metrics(y, preds)


# ── Attention analysis ────────────────────────────────────────────────────────

def token_attention(text, model, tok):
    """Per-token importance from DistilBERT's self-attention.

    Takes the attention that the [CLS] token (the one used for classification)
    pays to every other token, averaged over all 6 layers and 12 heads.
    [CLS] and [SEP] are dropped and the remaining scores are renormalised
    to sum to 1, so tokens can be compared within an article.
    """
    enc = tok(text, max_length=MAX_LEN, truncation=True, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        out = model(**enc, output_attentions=True)

    # out.attentions: tuple of 6 tensors, each (batch, heads, seq, seq)
    att = torch.stack(out.attentions).squeeze(1)   # (layers, heads, seq, seq)
    att = att.mean(dim=(0, 1))                      # (seq, seq)
    cls_att = att[0].cpu().numpy()                  # attention from [CLS] to each token

    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
    keep = [i for i, t in enumerate(tokens) if t not in ("[CLS]", "[SEP]")]
    tokens = [tokens[i] for i in keep]
    scores = cls_att[keep]
    scores = scores / scores.sum()
    return tokens, scores


def plot_attention(text, model, tok, title, n_tokens=40):
    """Heatmap of attention over the first n_tokens of an article."""
    tokens, scores = token_attention(text, model, tok)
    tokens, shown = tokens[:n_tokens], scores[:n_tokens]

    plt.figure(figsize=(max(10, len(tokens) * 0.35), 2.2))
    sns.heatmap(shown[np.newaxis, :], cmap="Reds", cbar=True,
                xticklabels=tokens, yticklabels=False)
    plt.xticks(rotation=70, ha="right", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    fname = "attention_" + re.sub(r"[^\w]", "_", title.lower()) + ".png"
    plt.savefig(fname, dpi=150)
    plt.show()

    top = np.argsort(scores)[::-1][:10]
    print(f"\n[Attention] {title} — top 10 tokens:")
    for i in top:
        print(f"  {tokens[i]:<15} {scores[i]:.4f}")


def reuters_attention_ratio(texts, model, tok, n_articles=200):
    """How much more attention does 'reuters' get than an average token?

    For each article containing the 'reuters' token, divides the attention
    on that token by the mean attention of all other tokens in the article.
    Returns the average ratio across articles (1.0 = no special attention).
    """
    ratios = []
    for text in texts[:n_articles]:
        tokens, scores = token_attention(text, model, tok)
        idx = [i for i, t in enumerate(tokens) if t == "reuters"]
        others = [s for i, s in enumerate(scores) if i not in idx]
        if idx and others:
            ratios.append(scores[idx].mean() / np.mean(others))

    if not ratios:
        print("\n[Attention] No articles containing 'reuters' found.")
        return None
    ratio = float(np.mean(ratios))
    print(f"\n[Attention] 'reuters' receives {ratio:.1f}x the attention of an "
          f"average token (mean over {len(ratios)} articles)")
    return ratio


def attention_analysis(model, tok, X_test, y_test, X_wf):
    """Compare attention on real ISOT articles vs WELFake articles."""
    print("\n[Attention analysis]")
    real_isot = [x for x, y in zip(X_test, y_test) if y == 0]

    plot_attention(real_isot[0], model, tok, "ISOT real article")
    plot_attention(X_wf[0], model, tok, "WELFake article")
    reuters_attention_ratio(real_isot, model, tok)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _metrics(y, preds):
    return {
        "accuracy":  accuracy_score(y, preds),
        "f1":        f1_score(y, preds, average="weighted"),
        "precision": precision_score(y, preds, average="weighted"),
        "recall":    recall_score(y, preds, average="weighted"),
    }


def _print_metrics(y, preds, label):
    r = _metrics(y, preds)
    print(f"\n{label}")
    print(f"  Accuracy:  {r['accuracy']:.4f}")
    print(f"  F1:        {r['f1']:.4f}")
    print(f"  Precision: {r['precision']:.4f}")
    print(f"  Recall:    {r['recall']:.4f}")
    print(classification_report(y, preds, target_names=["Real", "Fake"]))


def _confusion_matrix(y, preds, title):
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    fname = re.sub(r"[^\w]", "_", title.lower()) + ".png"
    plt.savefig(fname, dpi=150)
    plt.show()


def plot_comparison(results):
    labels   = list(results.keys())
    accuracy = [r["accuracy"] for r in results.values()]
    f1_vals  = [r["f1"]       for r in results.values()]
    x, w     = np.arange(len(labels)), 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w/2, accuracy, w, label="Accuracy",  color="steelblue",  alpha=0.85)
    b2 = ax.bar(x + w/2, f1_vals,  w, label="F1 Score",  color="darkorange", alpha=0.85)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + w/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: In-Domain (ISOT) vs Cross-Domain (WELFake)")
    ax.axvline(x=1.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    plt.show()


def predict(text, vec, clf, trainer, tok):
    cleaned = clean_text(text)

    proba    = clf.predict_proba(vec.transform([cleaned]))[0]
    bl_label = "FAKE" if proba[1] > 0.5 else "REAL"

    enc = tok(cleaned, max_length=MAX_LEN, padding="max_length",
              truncation=True, return_tensors="pt")
    trainer.model.eval()
    with torch.no_grad():
        logits = trainer.model(**enc).logits[0]
    probs      = torch.softmax(logits, dim=0)
    bert_label = "FAKE" if torch.argmax(probs).item() == 1 else "REAL"

    print(f"\n[Demo] {text[:100]}...")
    print(f"  Baseline   → {bl_label}  ({max(proba)*100:.1f}%)")
    print(f"  DistilBERT → {bert_label}  ({probs.max().item()*100:.1f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Fake News Detection — AI Lab 2025/2026")
    print("Ilayda Esma Asaroglu (2111742) · Liluz Atan (2186522)\n")

    # Load and prepare data
    df    = preprocess(load_isot())
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Train models
    vec, clf           = train_baseline(X_train, y_train, X_val, y_val)
    trainer, tok       = train_distilbert(X_train, y_train, X_val, y_val)

    # In-domain evaluation
    r_bl_isot   = eval_baseline(vec, clf, X_test, y_test, "ISOT Test")
    r_bert_isot = eval_distilbert(trainer, tok, X_test, y_test, "ISOT Test")

    # Cross-domain evaluation
    df_wf       = preprocess(load_welfake())
    X_wf, y_wf  = df_wf["text"].tolist(), df_wf["label"].tolist()
    r_bl_wf     = eval_baseline(vec, clf, X_wf, y_wf, "WELFake")
    r_bert_wf   = eval_distilbert(trainer, tok, X_wf, y_wf, "WELFake")

    # Why does performance drop? Look at what DistilBERT attends to
    attention_analysis(trainer.model, tok, X_test, y_test, X_wf)

    # Comparison chart
    plot_comparison({
        "Baseline (ISOT)":      r_bl_isot,
        "DistilBERT (ISOT)":    r_bert_isot,
        "Baseline (WELFake)":   r_bl_wf,
        "DistilBERT (WELFake)": r_bert_wf,
    })

    # Summary
    print("\n" + "="*58)
    print(f"  {'Model':<24} {'Dataset':<14} {'Acc':>7}  {'F1':>7}")
    print("  " + "-"*54)
    for name, ds, r in [
        ("TF-IDF + LR",  "ISOT Test", r_bl_isot),
        ("DistilBERT",   "ISOT Test", r_bert_isot),
        ("TF-IDF + LR",  "WELFake",   r_bl_wf),
        ("DistilBERT",   "WELFake",   r_bert_wf),
    ]:
        print(f"  {name:<24} {ds:<14} {r['accuracy']:>7.4f}  {r['f1']:>7.4f}")
    drop_bl   = r_bl_isot["accuracy"]   - r_bl_wf["accuracy"]
    drop_bert = r_bert_isot["accuracy"] - r_bert_wf["accuracy"]
    print("  " + "-"*54)
    print(f"  Drop (Baseline):   -{drop_bl:.4f}  ({drop_bl*100:.1f} pp)")
    print(f"  Drop (DistilBERT): -{drop_bert:.4f}  ({drop_bert*100:.1f} pp)")
    print("="*58)

    # Demo
    predict(
        "WASHINGTON (Reuters) - The Senate passed a bipartisan "
        "infrastructure bill on Tuesday with strong support from both parties.",
        vec, clf, trainer, tok
    )

    # Save model
    os.makedirs("./model", exist_ok=True)
    trainer.save_model("./model")
    tok.save_pretrained("./model")
    print("\nModel saved to ./model/")


if __name__ == "__main__":
    main()
