# ============================================================
#  SentiScope — NLP Sentiment Analyzer
#  Author : Vighan Raj Verma (@Vighan-coder)
#  GitHub : https://github.com/Vighan-coder/SentiScope
# ============================================================
#
#  SETUP:
#    pip install transformers torch datasets scikit-learn pandas
#                matplotlib bertopic sentence-transformers fastapi uvicorn
#
#  RUN TRAINING:
#    python sentiscope.py --mode train
#
#  RUN API:
#    python sentiscope.py --mode api
#
#  QUICK INFERENCE:
#    python sentiscope.py --mode demo
# ============================================================

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns


# ════════════════════════════════════════════════════════════
#  1. SYNTHETIC TWEET DATASET
# ════════════════════════════════════════════════════════════
POSITIVE_TEMPLATES = [
    "Absolutely love this! {}",
    "This is amazing, {} works perfectly!",
    "Best experience ever with {}. Highly recommend!",
    "So happy with {} today 😊",
    "Great job {} — keep it up!",
    "{} exceeded all my expectations!",
    "Fantastic product from {}. Will buy again!",
]
NEGATIVE_TEMPLATES = [
    "Really disappointed with {}. Never again.",
    "{} is the worst experience I've had.",
    "Terrible service from {}. Avoid at all costs.",
    "So frustrated with {} right now 😡",
    "{} broke after one day. Total waste of money.",
    "Cannot believe how bad {} treated their customers.",
    "Worst decision ever choosing {}.",
]
NEUTRAL_TEMPLATES = [
    "Just tried {} for the first time. It was okay.",
    "{} is neither good nor bad, just average.",
    "Used {} today. Nothing special to report.",
    "Ordered from {}. Still waiting for my package.",
    "{} updated their app. Some things changed.",
    "Had a meeting about {}. Discussed various topics.",
    "{} announced new features. Time will tell.",
]
BRANDS = ["Apple","Samsung","Nike","Adidas","Tesla","Google","Amazon",
          "Netflix","Spotify","Twitter","OpenAI","Meta","Microsoft"]


def generate_tweets(n=3000, seed=42):
    np.random.seed(seed)
    rows = []
    for _ in range(n):
        label = np.random.choice([0, 1, 2], p=[0.35, 0.45, 0.20])
        brand = np.random.choice(BRANDS)
        if label == 2:
            tmpl = np.random.choice(POSITIVE_TEMPLATES)
        elif label == 0:
            tmpl = np.random.choice(NEGATIVE_TEMPLATES)
        else:
            tmpl = np.random.choice(NEUTRAL_TEMPLATES)
        text = tmpl.format(brand)
        # Add some noise
        if np.random.rand() < 0.3:
            text += " " + np.random.choice(["#tech","#review","#brand","#product","#opinion"])
        rows.append({"text": text, "label": label, "brand": brand})
    df = pd.DataFrame(rows)
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    df["sentiment"] = df["label"].map(label_map)
    return df


# ════════════════════════════════════════════════════════════
#  2. BERT FINE-TUNING
# ════════════════════════════════════════════════════════════
def train_bert(df: pd.DataFrame):
    from transformers import (BertTokenizer, BertForSequenceClassification,
                              TrainingArguments, Trainer)
    from datasets import Dataset as HFDataset
    import torch

    print("[SentiScope] Loading BERT tokenizer …")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_df, val_df = train_test_split(df, test_size=0.15,
                                        stratify=df["label"], random_state=42)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True,
                         padding="max_length", max_length=128)

    train_ds = HFDataset.from_pandas(train_df[["text","label"]].reset_index(drop=True))
    val_ds   = HFDataset.from_pandas(val_df[["text","label"]].reset_index(drop=True))
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds   = val_ds.map(tokenize,   batched=True)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=3)

    args = TrainingArguments(
        output_dir="./sentiscope_bert",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(model=model, args=args,
                      train_dataset=train_ds, eval_dataset=val_ds)
    print("[SentiScope] Fine-tuning BERT …")
    trainer.train()
    trainer.save_model("./sentiscope_bert_final")
    print("[Saved] ./sentiscope_bert_final")
    return model, tokenizer, val_df


# ════════════════════════════════════════════════════════════
#  3. LIGHTWEIGHT FALLBACK (DistilBERT zero-shot — no training)
# ════════════════════════════════════════════════════════════
def zero_shot_predict(texts):
    from transformers import pipeline
    clf = pipeline("zero-shot-classification",
                   model="cross-encoder/nli-distilroberta-base")
    labels = ["positive sentiment", "negative sentiment", "neutral sentiment"]
    results = []
    for text in texts:
        out = clf(text, candidate_labels=labels)
        best = out["labels"][0].split()[0].capitalize()
        results.append(best)
    return results


# ════════════════════════════════════════════════════════════
#  4. BERTOPIC CLUSTERING
# ════════════════════════════════════════════════════════════
def run_topic_modeling(df: pd.DataFrame):
    try:
        from bertopic import BERTopic
        model  = BERTopic(language="english", calculate_probabilities=True,
                          verbose=True, nr_topics=10)
        topics, probs = model.fit_transform(df["text"].tolist())
        df["topic"] = topics
        info = model.get_topic_info()
        print("\n── Top Topics ─────────────────────────────────────")
        print(info[["Topic","Count","Name"]].head(10).to_string(index=False))
        model.save("sentiscope_bertopic")
        print("[Saved] sentiscope_bertopic/")
        return model, df
    except ImportError:
        print("[SentiScope] BERTopic not installed — skipping topic modeling.")
        return None, df


# ════════════════════════════════════════════════════════════
#  5. VISUALISATIONS
# ════════════════════════════════════════════════════════════
def plot_distribution(df: pd.DataFrame):
    counts = df["sentiment"].value_counts()
    colors = {"Positive":"#7cff67","Neutral":"#B19EEF","Negative":"#ff6b6b"}
    plt.figure(figsize=(7,4))
    bars = plt.bar(counts.index, counts.values,
                   color=[colors.get(s,"grey") for s in counts.index])
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                 str(val), ha="center", fontsize=11)
    plt.title("Sentiment Distribution — SentiScope")
    plt.tight_layout()
    plt.savefig("sentiment_distribution.png", dpi=150)
    plt.show()
    print("[Saved] sentiment_distribution.png")


def plot_brand_heatmap(df: pd.DataFrame):
    pivot = pd.crosstab(df["brand"], df["sentiment"])
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGn")
    plt.title("Sentiment Ratio by Brand")
    plt.tight_layout()
    plt.savefig("brand_sentiment_heatmap.png", dpi=150)
    plt.show()
    print("[Saved] brand_sentiment_heatmap.png")


# ════════════════════════════════════════════════════════════
#  6. FASTAPI INFERENCE SERVER
# ════════════════════════════════════════════════════════════
def run_api():
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
    from transformers import pipeline

    app = FastAPI(title="SentiScope API")
    clf = pipeline("text-classification",
                   model="distilbert-base-uncased-finetuned-sst-2-english")

    class TextInput(BaseModel):
        text: str

    @app.get("/")
    def root():
        return {"message": "SentiScope API is running 🚀"}

    @app.post("/predict")
    def predict(payload: TextInput):
        result = clf(payload.text)[0]
        return {
            "text"      : payload.text,
            "sentiment" : result["label"],
            "confidence": round(result["score"], 4)
        }

    print("[SentiScope] API running at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ════════════════════════════════════════════════════════════
#  7. DEMO MODE — quick inference on sample sentences
# ════════════════════════════════════════════════════════════
def demo():
    samples = [
        "I absolutely love this product, it works perfectly!",
        "This is the worst service I have ever experienced.",
        "The package arrived today. It's fine I guess.",
        "Tesla's new update is incredible! So smooth 😍",
        "Google's new policy is frustrating and unfair.",
        "Amazon delivered my order. Nothing special.",
    ]
    print("\n── SentiScope Demo ────────────────────────────────")
    try:
        from transformers import pipeline
        clf = pipeline("text-classification",
                       model="distilbert-base-uncased-finetuned-sst-2-english")
        for s in samples:
            r = clf(s)[0]
            icon = "🟢" if r["label"] == "POSITIVE" else "🔴"
            print(f"{icon} [{r['label']:8s} {r['score']:.2f}]  {s}")
    except Exception as e:
        print(f"Transformers not available ({e}). Printing samples only.")
        for s in samples:
            print(f"  → {s}")
    print()


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","api","demo","eda"],
                        default="demo")
    args = parser.parse_args()

    df = generate_tweets(n=3000)
    print(f"[SentiScope] Dataset: {len(df)} tweets  |  "
          f"Pos={( df.sentiment=='Positive').sum()}  "
          f"Neg={(df.sentiment=='Negative').sum()}  "
          f"Neu={(df.sentiment=='Neutral').sum()}")

    if args.mode == "train":
        train_bert(df)
        run_topic_modeling(df)
        plot_distribution(df)
        plot_brand_heatmap(df)

    elif args.mode == "api":
        run_api()

    elif args.mode == "eda":
        plot_distribution(df)
        plot_brand_heatmap(df)

    else:  # demo
        demo()
        plot_distribution(df)
        plot_brand_heatmap(df)

    print("[SentiScope] Done!")