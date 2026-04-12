"""
ML Version 3 — Pure Word2Vec (Unweighted Mean)
================================================
Each document is represented as a single 300-dim vector computed by
taking the simple unweighted mean of all in-vocabulary GloVe word
vectors. Every word contributes equally regardless of importance.
Fastest and simplest of the three approaches.

Classifiers:
  - Support Vector Machine (RBF kernel)
  - Logistic Regression

Unknown class: if the model's confidence (max class probability) is
below CONFIDENCE_THRESHOLD the prediction is overridden to "Unknown"
instead of guessing the closest class.

Input CSVs must have columns: text, label
Numeric features are concatenated onto the embedding before training.
"""

import csv
import numpy as np
import gensim.downloader as gensim_api
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

TRAIN_CSV            = "train.csv"
TEST_CSV             = "test.csv"
VECTOR_DIM           = 300
CONFIDENCE_THRESHOLD = 0.6   # If max class probability < this, predict "Unknown"
                              # Raise to flag more documents as Unknown (stricter)
                              # Lower to only flag very uncertain predictions (looser)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_csv(path):
    texts, labels, numeric = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(row["label"].strip())
            numeric.append([
                float(row.get("table_row_count",  0) or 0),
                float(row.get("table_text_ratio", 0) or 0),
                float(row.get("avg_cells_per_row",0) or 0),
            ])
    return texts, labels, np.array(numeric)

def normalise_label(l):
    if "state eligibility" in l.lower():
        return "State Eligibility"
    return l

print("=" * 60)
print("ML VERSION 3 — PURE WORD2VEC (UNWEIGHTED MEAN)")
print("=" * 60)
print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD} "
      f"(predictions below this become 'Unknown')")

print("\n[1/4] Loading data...")
train_texts, train_labels, train_numeric = load_csv(TRAIN_CSV)
test_texts,  test_labels,  test_numeric  = load_csv(TEST_CSV)
train_labels = [normalise_label(l) for l in train_labels]
test_labels  = [normalise_label(l) for l in test_labels]
print(f"  Train: {len(train_texts)} samples")
print(f"  Test:  {len(test_texts)} samples")
print(f"  Classes: {sorted(set(train_labels))}")

# ---------------------------------------------------------------------------
# 2. Load Word2Vec
# ---------------------------------------------------------------------------

print("\n[2/4] Loading Word2Vec model (GloVe 300-dim, ~1.6 GB)...")
print("  This downloads once to ~/.cache/gensim-data/ — may take a few minutes.")
w2v = gensim_api.load("glove-wiki-gigaword-300")
print("  Word2Vec ready.")

# ---------------------------------------------------------------------------
# 3. Build feature vectors
# ---------------------------------------------------------------------------

def clean_tokens(text):
    return [t.lower() for t in text.split() if t.isalpha() and len(t) > 2]

def w2v_vector(text):
    """300-dim unweighted mean of GloVe vectors, L2-normalised."""
    tokens = clean_tokens(text)
    vecs   = [w2v[t] for t in tokens if t in w2v]
    if not vecs:
        return np.zeros(VECTOR_DIM)
    vec = np.mean(vecs, axis=0)
    return vec / (np.linalg.norm(vec) + 1e-10)

def build_features(texts, numeric):
    embeddings = np.array([w2v_vector(t) for t in texts])
    return np.hstack([embeddings, numeric])

print("\n[3/4] Building feature vectors...")
X_train = build_features(train_texts, train_numeric)
X_test  = build_features(test_texts,  test_numeric)

oov_rates = []
for text in train_texts + test_texts:
    tokens = clean_tokens(text)
    if tokens:
        oov_rates.append(sum(1 for t in tokens if t not in w2v) / len(tokens))

le = LabelEncoder()
y_train = le.fit_transform(train_labels)
y_test  = le.transform(test_labels)

print(f"  Train feature matrix: {X_train.shape}")
print(f"  Test  feature matrix: {X_test.shape}")
print(f"  Breakdown: 300 Word2Vec dims (unweighted) + 3 numeric dims = {X_train.shape[1]} total")
print(f"  Avg OOV rate: {np.mean(oov_rates):.1%}")

# ---------------------------------------------------------------------------
# 4. Train and evaluate
# ---------------------------------------------------------------------------

print("\n[4/4] Training and evaluating classifiers...\n")

def predict_with_confidence(model, X, le, threshold):
    """
    Returns (label_strings, confidences) where any prediction whose
    max class probability is below `threshold` is replaced with 'Unknown'.
    """
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

def evaluate(name, model, X_tr, y_tr, X_te, y_te, le, threshold):
    model.fit(X_tr, y_tr)
    pred_labels, confidences = predict_with_confidence(model, X_te, le, threshold)
    actual_labels = le.inverse_transform(y_te)

    committed     = [(p, a) for p, a in zip(pred_labels, actual_labels) if p != "Unknown"]
    n_unknown     = pred_labels.count("Unknown")
    acc_committed = sum(p == a for p, a in committed) / len(committed) if committed else 0.0
    acc_overall   = sum(p == a for p, a in zip(pred_labels, actual_labels)) / len(pred_labels)

    known_preds   = [p for p in pred_labels if p != "Unknown"]
    known_actuals = [a for p, a in zip(pred_labels, actual_labels) if p != "Unknown"]
    all_classes   = sorted(set(actual_labels))

    print(f"{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")
    print(f"  Overall accuracy  : {acc_overall:.2%}  ({sum(p==a for p,a in zip(pred_labels,actual_labels))}/{len(pred_labels)})")
    print(f"  Committed accuracy: {acc_committed:.2%}  (excluding {n_unknown} Unknown predictions)")
    print(f"  Unknown count     : {n_unknown}/{len(pred_labels)} documents below confidence threshold\n")

    if known_preds:
        report = classification_report(known_actuals, known_preds,
                                       labels=all_classes, zero_division=0)
        print("  Classification Report (committed predictions only):")
        for line in report.splitlines():
            print("    " + line)

    print("\n  Per-sample predictions:")
    for i, (pred, actual, conf) in enumerate(zip(pred_labels, actual_labels, confidences)):
        if pred == "Unknown":
            status = "?"
        elif pred == actual:
            status = "✓"
        else:
            status = "✗"
        print(f"    [{status}] Sample {i+1:02d}  conf={conf:.2f}  "
              f"predicted: {pred:<35} actual: {actual}")
    print()

# Both classifiers are wrapped in a Pipeline so StandardScaler is fitted
# on X_train only and applied to X_test — no data leakage.
evaluate(
    "SVM (RBF kernel) + StandardScaler",
    Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)),
    ]),
    X_train, y_train, X_test, y_test, le, CONFIDENCE_THRESHOLD
)

evaluate(
    "Logistic Regression + StandardScaler",
    Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(max_iter=1000, C=1.0)),
    ]),
    X_train, y_train, X_test, y_test, le, CONFIDENCE_THRESHOLD
)

print("=" * 60)
print("Done.")