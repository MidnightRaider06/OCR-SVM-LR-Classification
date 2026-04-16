"""
ML Version 1 — TF-IDF  (with K-Fold Cross-Validation)
=======================================================
Vectorizes document text using TF-IDF fitted on a background corpus
(20 Newsgroups, ~11,000 articles) so every document always produces
a fixed 500-dim vector with real IDF weights.

Because the training set is small (100 samples), we use Stratified
K-Fold cross-validation instead of a single train/test split.
This gives a much more reliable accuracy estimate by training and
evaluating on every sample rather than a fixed held-out set.

What K-Fold does:
  - Splits the training data into K equal folds (default K=5)
  - Trains K separate models, each time holding out one fold as the
    test set and training on the remaining K-1 folds
  - Reports mean ± std accuracy across all K folds
  - A final model is then trained on ALL training data and evaluated
    on the held-out test CSV for a real-world check

Classifiers:
  - Support Vector Machine (linear kernel)
  - Logistic Regression  ← also saved to disk

Unknown class: if the model's confidence (max class probability) is
below CONFIDENCE_THRESHOLD the prediction is overridden to "Unknown".

Input CSVs must have columns: text, label
"""

import csv
import os
import numpy as np
import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, make_scorer
)


MODEL_SAVE_PATH1      = "../backend/lr_model_v1.joblib"
MODEL_SAVE_PATH2      = "../backend/svm_model_v1.joblib"
TRAIN_CSV            = "train.csv"
TEST_CSV             = "test.csv"
CONFIDENCE_THRESHOLD = 0.6   # below this → "Unknown"
N_FOLDS              = 5     # number of CV folds — increase if you have more data

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_csv(path):
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(row["label"].strip())
    return texts, labels

def normalise_label(l):
    if "state eligibility" in l.lower():
        return "State Eligibility"
    return l

print("=" * 60)
print("ML VERSION 1 — TF-IDF  (K-Fold Cross-Validation)")
print("=" * 60)
print(f"  Folds            : {N_FOLDS}")
print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")

print("\n[1/6] Loading data...")
train_texts, train_labels = load_csv(TRAIN_CSV)
test_texts,  test_labels  = load_csv(TEST_CSV)
train_labels = [normalise_label(l) for l in train_labels]
test_labels  = [normalise_label(l) for l in test_labels]
print(f"  Train: {len(train_texts)} samples")
print(f"  Test:  {len(test_texts)} samples")
print(f"  Classes: {sorted(set(train_labels))}")

# ---------------------------------------------------------------------------
# 2. Fit TF-IDF on background corpus
#
# IMPORTANT: The TF-IDF vectorizer is fitted ONCE here on the background
# corpus and never re-fitted inside the CV loop. This means the vocabulary
# and IDF weights are fixed — we only call transform() per fold, not
# fit_transform(). This is the correct approach because:
#   a) It prevents data leakage (fold test data never influences the vocab)
#   b) The vectorizer sees the same 500-term vocabulary every fold,
#      making fold results directly comparable
# ---------------------------------------------------------------------------

print("\n[2/6] Fitting TF-IDF on background corpus (~11k news articles)...")
corpus = fetch_20newsgroups(subset="train").data
tfidf  = TfidfVectorizer(
    stop_words="english",
    max_features=500,
    ngram_range=(1, 1),
    min_df=2,
)
tfidf.fit(corpus)
print(f"  Fitted on {len(corpus)} docs — vocabulary locked at {len(tfidf.vocabulary_)} terms")

# ---------------------------------------------------------------------------
# 3. Build full feature matrix from training data
# ---------------------------------------------------------------------------

print("\n[3/6] Building feature vectors...")

def build_features(texts):
    vecs = tfidf.transform(texts).toarray()     # (n, 500) — transform only
    return sk_normalize(vecs, norm="l2")        # L2-normalise rows

X_train = build_features(train_texts)
X_test  = build_features(test_texts)

le = LabelEncoder()
y_train = le.fit_transform(train_labels)
y_test  = le.transform(test_labels)

print(f"  Train feature matrix: {X_train.shape}")
print(f"  Test  feature matrix: {X_test.shape}")

# ---------------------------------------------------------------------------
# 4. K-Fold cross-validation
#
# StratifiedKFold preserves the class distribution in every fold.
# This matters here because some classes have only 20 samples — a random
# split could put all of one class in the test fold, making scores useless.
#
# For each fold we:
#   1. Split X_train / y_train into fold_train and fold_val
#   2. Fit the model on fold_train
#   3. Predict on fold_val and record accuracy + per-class scores
# Then report mean ± std across all K folds.
# ---------------------------------------------------------------------------

print(f"\n[4/6] {N_FOLDS}-Fold Stratified Cross-Validation...\n")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

def predict_with_confidence(model, X, le, threshold):
    probs      = model.predict_proba(X)
    max_probs  = probs.max(axis=1)
    raw_preds  = probs.argmax(axis=1)
    labels_out = []
    for pred_idx, conf in zip(raw_preds, max_probs):
        if conf < threshold:
            labels_out.append("Unknown")
        else:
            labels_out.append(le.inverse_transform([pred_idx])[0])
    return labels_out, max_probs

def run_cv(name, make_model):
    """
    Runs K-Fold CV and prints per-fold + summary results.
    Returns the model trained on ALL training data (for final evaluation).
    """
    print(f"{'─' * 60}")
    print(f"  {name}  —  {N_FOLDS}-Fold CV")
    print(f"{'─' * 60}")

    fold_accs        = []
    fold_committed   = []
    fold_unknown_pct = []
    all_classes      = sorted(le.classes_)

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        model = make_model()
        model.fit(X_tr, y_tr)

        pred_labels, confidences = predict_with_confidence(model, X_val, le, CONFIDENCE_THRESHOLD)
        actual_labels = le.inverse_transform(y_val)

        n_unknown     = pred_labels.count("Unknown")
        committed     = [(p, a) for p, a in zip(pred_labels, actual_labels) if p != "Unknown"]
        acc_overall   = sum(p == a for p, a in zip(pred_labels, actual_labels)) / len(pred_labels)
        acc_committed = sum(p == a for p, a in committed) / len(committed) if committed else 0.0

        fold_accs.append(acc_overall)
        fold_committed.append(acc_committed)
        fold_unknown_pct.append(n_unknown / len(pred_labels))

        print(f"  Fold {fold_idx}:  acc={acc_overall:.2%}  "
              f"committed={acc_committed:.2%}  "
              f"unknown={n_unknown}/{len(pred_labels)}")

    print(f"\n  ── CV Summary ──────────────────────────────────────")
    print(f"  Overall   accuracy  mean={np.mean(fold_accs):.2%}  "
          f"std=±{np.std(fold_accs):.2%}  "
          f"min={np.min(fold_accs):.2%}  max={np.max(fold_accs):.2%}")
    print(f"  Committed accuracy  mean={np.mean(fold_committed):.2%}  "
          f"std=±{np.std(fold_committed):.2%}")
    print(f"  Unknown rate        mean={np.mean(fold_unknown_pct):.1%}  "
          f"(avg docs flagged as Unknown per fold)")

    # ── Final model: train on ALL training data, evaluate on test CSV ──
    print(f"\n  ── Final model (trained on all {len(X_train)} train samples) ──")
    final_model = make_model()
    final_model.fit(X_train, y_train)

    pred_labels, confidences = predict_with_confidence(final_model, X_test, le, CONFIDENCE_THRESHOLD)
    actual_labels = le.inverse_transform(y_test)

    n_unknown     = pred_labels.count("Unknown")
    committed     = [(p, a) for p, a in zip(pred_labels, actual_labels) if p != "Unknown"]
    acc_overall   = sum(p == a for p, a in zip(pred_labels, actual_labels)) / len(pred_labels)
    acc_committed = sum(p == a for p, a in committed) / len(committed) if committed else 0.0

    print(f"  Test accuracy  : {acc_overall:.2%}  ({sum(p==a for p,a in zip(pred_labels,actual_labels))}/{len(pred_labels)})")
    print(f"  Committed acc  : {acc_committed:.2%}  (excl. {n_unknown} Unknown)")
    print(f"  Unknown count  : {n_unknown}/{len(pred_labels)}\n")

    if [p for p in pred_labels if p != "Unknown"]:
        known_preds   = [p for p in pred_labels if p != "Unknown"]
        known_actuals = [a for p, a in zip(pred_labels, actual_labels) if p != "Unknown"]
        report = classification_report(known_actuals, known_preds,
                                       labels=all_classes, zero_division=0)
        print("  Classification Report (committed predictions only):")
        for line in report.splitlines():
            print("    " + line)

    print("\n  Per-sample test predictions:")
    for i, (pred, actual, conf) in enumerate(zip(pred_labels, actual_labels, confidences)):
        status = "?" if pred == "Unknown" else ("✓" if pred == actual else "✗")
        print(f"    [{status}] Sample {i+1:02d}  conf={conf:.2f}  "
              f"predicted: {pred:<35} actual: {actual}")
    print()

    return final_model

svm_model = run_cv(
    "SVM (linear kernel)",
    lambda: SVC(kernel="linear", C=1.0, probability=True),
)

lr_model = run_cv(
    "Logistic Regression",
    lambda: LogisticRegression(max_iter=1000, C=1.0),
)

# ---------------------------------------------------------------------------
# 5. Save the final Logistic Regression model
#
# The saved model was trained on ALL training data (not just one fold),
# which gives the best possible weights for production use.
# The CV results above give an honest estimate of how well it will
# generalise — use those numbers, not the test CSV accuracy alone.
# ---------------------------------------------------------------------------

print("[5/6] Saving Logistic Regression model...")
joblib.dump(
    {"model": lr_model, "tfidf": tfidf, "label_encoder": le},
    MODEL_SAVE_PATH1
)
print(f"  Saved to   : {MODEL_SAVE_PATH1}")
print(f"  Contains   : LogisticRegression + TfidfVectorizer + LabelEncoder")
print(f"  Trained on : all {len(X_train)} training samples")
print()


print("[6/6] Saving Logistic Regression model...")
joblib.dump(
    {"model": svm_model, "tfidf": tfidf, "label_encoder": le},
    MODEL_SAVE_PATH2
)

print(f"  Saved to   : {MODEL_SAVE_PATH2}")
print(f"  Contains   : LogisticRegression + TfidfVectorizer + LabelEncoder")
print(f"  Trained on : all {len(X_train)} training samples")
print()



print("=" * 60)
print("Done.")
print()
print("Interpreting CV results:")
print("  mean accuracy — best single number to report")
print("  std           — how stable the model is across folds")
print("                  large std = model is sensitive to which")
print("                  samples it sees (common with small datasets)")
print("  committed acc — accuracy on docs the model was confident")
print("                  about (excludes Unknown predictions)")