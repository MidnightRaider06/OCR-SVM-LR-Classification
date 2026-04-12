from __future__ import annotations

import io
import os
import re
import tempfile

import numpy as np
from PIL import Image
import pytesseract
import joblib
from bs4 import BeautifulSoup
from paddleocr import TableRecognitionPipelineV2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.datasets import fetch_20newsgroups
import gensim.downloader as gensim_api

# ---------------------------------------------------------------------------
# Engine — loaded once at import time
# ---------------------------------------------------------------------------

table_engine = TableRecognitionPipelineV2(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    layout_detection_model_name="PP-DocLayout-L",
    wired_table_structure_recognition_model_name="SLANet_plus",
    wireless_table_structure_recognition_model_name="SLANet_plus",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    enable_mkldnn=True,
    cpu_threads=8,
)

# ---------------------------------------------------------------------------
# NLP models — loaded and fitted once at import time
# ---------------------------------------------------------------------------

print("Fitting TF-IDF on background corpus...")
_background_corpus = fetch_20newsgroups(subset="train").data
_tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=500,
    ngram_range=(1, 1),
    min_df=2,
)
_tfidf.fit(_background_corpus)
print(f"TF-IDF vocabulary size: {len(_tfidf.vocabulary_)} terms")

print("Loading Word2Vec model...")
_w2v = gensim_api.load("glove-wiki-gigaword-300")
VECTOR_DIM = 300
print("Models ready.")

# ---------------------------------------------------------------------------
# Saved classifiers — both loaded once at import time
#
# svm_model_v1.joblib — SVC linear kernel  (CV mean=100% ± 0%)
# lr_model_v1.joblib  — LogisticRegression (CV mean=91%  ± 8.6%)
#
# Each file contains: {"model": ..., "tfidf": ..., "label_encoder": ...}
# We use the tfidf/label_encoder from each saved file so each classifier
# always sees the exact vocabulary it was trained on.
# ---------------------------------------------------------------------------

_CONFIDENCE_THRESHOLD = 0.6

_SVM_PATH = os.path.join(os.path.dirname(__file__), "svm_model_v1.joblib")
_LR_PATH  = os.path.join(os.path.dirname(__file__), "lr_model_v1.joblib")

_svm_classifier  = None
_svm_tfidf       = None
_svm_label_enc   = None

_lr_classifier   = None
_lr_tfidf        = None
_lr_label_enc    = None

if os.path.exists(_SVM_PATH):
    print(f"Loading SVM classifier from {_SVM_PATH}...")
    _s = joblib.load(_SVM_PATH)
    _svm_classifier = _s["model"]
    _svm_tfidf      = _s["tfidf"]
    _svm_label_enc  = _s["label_encoder"]
    print("SVM classifier ready.")
else:
    print(f"WARNING: {_SVM_PATH} not found. Run ml_v1_tfidf.py to generate it.")

if os.path.exists(_LR_PATH):
    print(f"Loading LR classifier from {_LR_PATH}...")
    _l = joblib.load(_LR_PATH)
    _lr_classifier = _l["model"]
    _lr_tfidf      = _l["tfidf"]
    _lr_label_enc  = _l["label_encoder"]
    print("LR classifier ready.")
else:
    print(f"WARNING: {_LR_PATH} not found. Run ml_v1_tfidf.py to generate it.")


# ---------------------------------------------------------------------------
# Shared prediction helper
# ---------------------------------------------------------------------------

def _run_prediction(classifier, clf_tfidf, clf_label_enc, text: str, features: dict) -> dict:
    """
    Builds a 503-dim feature vector and runs predict_proba on the given model.
    Returns label, confidence, and all class probabilities.
    Returns {"label": "unavailable"} if the classifier was not loaded.
    """
    if classifier is None:
        return {"label": "unavailable", "confidence": 0.0, "all_probs": {}}

    tfidf_vec = clf_tfidf.transform([text]).toarray()    # (1, 500)
    tfidf_vec = sk_normalize(tfidf_vec, norm="l2")
    numeric   = np.array([[
        float(features.get("table_row_count",  0) or 0),
        float(features.get("table_text_ratio", 0) or 0),
        float(features.get("avg_cells_per_row",0) or 0),
    ]])
    X = np.hstack([tfidf_vec, numeric])                  # (1, 503)

    probs       = classifier.predict_proba(X)[0]
    max_conf    = float(probs.max())
    pred_idx    = int(probs.argmax())
    class_names = clf_label_enc.classes_

    label = (
        clf_label_enc.inverse_transform([pred_idx])[0]
        if max_conf >= _CONFIDENCE_THRESHOLD
        else "Unknown"
    )

    return {
        "label":      label,
        "confidence": round(max_conf, 4),
        "all_probs":  {
            name: round(float(p), 4)
            for name, p in zip(class_names, probs)
        },
    }


# ---------------------------------------------------------------------------
# Public entry point — called by main.py
# ---------------------------------------------------------------------------

def run_ocr(
    file_bytes: bytes,
    *,
    filename: str | None = None,
    content_type: str | None = None,
    mode: str | None = None,
) -> dict:
    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tables, table_bboxes = _extract_tables(file_bytes)
    raw_text = _extract_raw_text(img, table_bboxes)

    all_rows = [row for table in tables for row in table["rows"]]

    text_word_count  = len(raw_text.split())
    table_word_count = sum(len(cell.split()) for row in all_rows for cell in row)
    total_word_count = text_word_count + table_word_count
    features = {
        "table_row_count":   len(all_rows),
        "table_text_ratio":  round(table_word_count / total_word_count, 4) if total_word_count > 0 else 0.0,
        "avg_cells_per_row": round(sum(len(row) for row in all_rows) / len(all_rows), 4) if all_rows else 0.0,
    }

    table_text      = " ".join(cell for row in all_rows for cell in row)
    flat_text       = " ".join(filter(None, [raw_text, table_text]))
    normalized_text = _normalize(flat_text)

    v1 = _version1_tfidf(normalized_text)
    v2 = _version2_tfidf_w2v(normalized_text)
    v3 = _version3_w2v(normalized_text)

    # Run both classifiers independently on the same text + features
    svm_prediction = _run_prediction(
        _svm_classifier, _svm_tfidf, _svm_label_enc, normalized_text, features
    )
    lr_prediction = _run_prediction(
        _lr_classifier, _lr_tfidf, _lr_label_enc, normalized_text, features
    )

    tables_html = [t["html"] for t in tables if t.get("html")]
    return {
        "text":        normalized_text,
        "features":    features,
        "tables_html": tables_html,
        "tfidf":       v1,
        "tfidf_w2v":   v2,
        "w2v":         v3,

        # SVM prediction (CV mean=100% ± 0% — more stable)
        # e.g. {"label": "Eligibility Evidence", "confidence": 0.94, "all_probs": {...}}
        "svm_prediction": svm_prediction,

        # LR prediction  (CV mean=91%  ± 8.6%)
        # e.g. {"label": "Eligibility Evidence", "confidence": 0.78, "all_probs": {...}}
        "lr_prediction":  lr_prediction,
    }


# ---------------------------------------------------------------------------
# VERSION 1 — TF-IDF
# ---------------------------------------------------------------------------

def _version1_tfidf(text: str, top_n: int = 20) -> dict:
    empty = {"keywords": [], "vector": [], "vocab_size": 0}
    if not text.strip():
        return empty
    try:
        matrix = _tfidf.transform([text])
        vocab  = _tfidf.get_feature_names_out()
        scores = matrix.toarray()[0]
        normed = sk_normalize(scores.reshape(1, -1), norm="l2")[0]
        ranked   = sorted(zip(vocab, scores), key=lambda x: x[1], reverse=True)
        keywords = [{"word": w, "score": round(float(s), 4)} for w, s in ranked[:top_n] if s > 0]
        return {
            "keywords":   keywords,
            "vector":     [round(float(v), 6) for v in normed],
            "vocab_size": int((scores > 0).sum()),
        }
    except ValueError:
        return empty


# ---------------------------------------------------------------------------
# VERSION 2 — TF-IDF weighted Word2Vec
# ---------------------------------------------------------------------------

def _version2_tfidf_w2v(text: str, top_n: int = 20) -> dict:
    tokens = _clean_tokens(text)
    empty  = {"keywords": [], "vector": None, "oov_rate": 1.0}
    if not tokens:
        return empty

    weights  = {}
    keywords = []
    try:
        mat    = _tfidf.transform([" ".join(tokens)])
        vocab  = _tfidf.get_feature_names_out()
        scores = mat.toarray()[0]
        weights  = dict(zip(vocab, scores))
        ranked   = sorted(zip(vocab, scores), key=lambda x: x[1], reverse=True)
        keywords = [{"word": w, "score": round(float(s), 4)} for w, s in ranked[:top_n] if s > 0]
    except ValueError:
        pass

    weighted_vecs = []
    total_weight  = 0.0
    oov_count     = 0

    for token in tokens:
        if token in _w2v:
            w = weights.get(token, 1.0)
            weighted_vecs.append(_w2v[token] * w)
            total_weight += w
        else:
            oov_count += 1

    oov_rate = round(oov_count / len(tokens), 4) if tokens else 1.0

    if not weighted_vecs or total_weight == 0:
        return {"keywords": keywords, "vector": None, "oov_rate": oov_rate}

    doc_vec = np.sum(weighted_vecs, axis=0) / total_weight
    doc_vec = doc_vec / (np.linalg.norm(doc_vec) + 1e-10)

    return {
        "keywords": keywords,
        "vector":   [round(float(v), 6) for v in doc_vec],
        "oov_rate": oov_rate,
    }


# ---------------------------------------------------------------------------
# VERSION 3 — Pure Word2Vec (unweighted mean)
# ---------------------------------------------------------------------------

def _version3_w2v(text: str) -> dict:
    tokens = _clean_tokens(text)
    empty  = {"tokens": [], "vector": None, "oov_rate": 1.0}
    if not tokens:
        return empty

    vecs      = []
    oov_count = 0

    for token in tokens:
        if token in _w2v:
            vecs.append(_w2v[token])
        else:
            oov_count += 1

    oov_rate = round(oov_count / len(tokens), 4)

    if not vecs:
        return {"tokens": tokens, "vector": None, "oov_rate": oov_rate}

    doc_vec = np.mean(vecs, axis=0)
    doc_vec = doc_vec / (np.linalg.norm(doc_vec) + 1e-10)

    return {
        "tokens":   tokens,
        "vector":   [round(float(v), 6) for v in doc_vec],
        "oov_rate": oov_rate,
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clean_tokens(text: str) -> list[str]:
    return [t.lower() for t in text.split() if t.isalpha() and len(t) > 2]


def _extract_tables(image_bytes: bytes):
    suffix = _sniff_suffix(image_bytes)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    try:
        result = table_engine.predict(tmp_path)
    finally:
        os.unlink(tmp_path)

    tables       = []
    table_bboxes = []

    for res in result:
        inner          = res.json.get("res", {})
        boxes          = inner.get("layout_det_res", {}).get("boxes", [])
        table_res_list = inner.get("table_res_list", [])

        table_boxes                 = [b for b in boxes if b.get("label") == "table"]
        table_boxes, table_res_list = _deduplicate_tables(table_boxes, table_res_list)

        for i, table_data in enumerate(table_res_list):
            bbox = table_boxes[i]["coordinate"] if i < len(table_boxes) else []
            if bbox:
                table_bboxes.append(bbox)
                tables.append({
                    "bbox": bbox,
                    "rows": _parse_table_rows(table_data),
                    "html": table_data.get("pred_html", ""),
                })

    return tables, table_bboxes


def _parse_table_rows(table_data: dict) -> list:
    html = table_data.get("pred_html", "")
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for row in soup.find_all("tr"):
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        cells = [c for c in cells if c]
        if cells:
            rows.append(cells)
    return rows


def _mask_table_regions(img: Image.Image, table_bboxes: list, padding: int = 5) -> Image.Image:
    img_np = np.array(img)
    for bbox in table_bboxes:
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(img_np.shape[1], int(x2) + padding)
        y2 = min(img_np.shape[0], int(y2) + padding)
        img_np[y1:y2, x1:x2] = 255
    return Image.fromarray(img_np)


def _extract_raw_text(img: Image.Image, table_bboxes: list) -> str:
    masked_img = _mask_table_regions(img, table_bboxes)
    word_data  = pytesseract.image_to_data(
        masked_img,
        config=r"--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT,
    )
    words = [
        word_data["text"][i].strip()
        for i in range(len(word_data["text"]))
        if word_data["text"][i].strip() and int(word_data["conf"][i]) > 40
    ]
    return " ".join(words)


def _iomin(a: list, b: list) -> float:
    x1           = max(a[0], b[0])
    y1           = max(a[1], b[1])
    x2           = min(a[2], b[2])
    y2           = min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a       = (a[2] - a[0]) * (a[3] - a[1])
    area_b       = (b[2] - b[0]) * (b[3] - b[1])
    min_area     = min(area_a, area_b)
    return intersection / min_area if min_area > 0 else 0.0


def _deduplicate_tables(table_boxes: list, table_res_list: list, iomin_threshold: float = 0.8):
    indexed = sorted(range(len(table_boxes)), key=lambda i: table_boxes[i].get("score", 0), reverse=True)
    keep    = []
    for i in indexed:
        coord = table_boxes[i]["coordinate"]
        if all(_iomin(coord, table_boxes[j]["coordinate"]) < iomin_threshold for j in keep):
            keep.append(i)
    keep.sort()
    return [table_boxes[i] for i in keep], [table_res_list[i] for i in keep]


def _normalize(text: str) -> str:
    text = re.sub(r'[^\x20-\x7E]+', ' ', text)
    text = re.sub(r' {2,}', ' ', text).strip()
    text = re.sub(r'\(?\b\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}\b', 'PHONE_NO', text)
    text = re.sub(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b', 'DATE', text)
    text = re.sub(r'\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b', 'DATE', text)
    return text


def _sniff_suffix(image_bytes: bytes) -> str:
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if image_bytes[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tiff"
    if image_bytes[:2] == b"BM":
        return ".bmp"
    return ".png"