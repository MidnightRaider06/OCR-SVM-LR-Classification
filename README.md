# Structured Document Pipeline

Minimal OCR demo for turning uploaded images and PDFs into raw OCR output, normalized JSON, and report-friendly document data.

## Stack

- Backend: FastAPI
- OCR: PaddleOCR / PP-StructureV3
- Frontend: React
- PDF export: jsPDF

## Layout

- `backend/`
  FastAPI app, OCR execution, table parsing, document normalization, and tests.
- `frontend/src/`
  React source for upload, preview, run state, and results display.
- `frontend/`
  Built static assets served by the app.
- `test_images/`
  Sample inputs for local testing.

## Flow

Upload files -> run OCR -> build raw document blocks -> normalize into structured JSON -> render results / export report.

1. Install dependencies:
```bash
   cd backend
   python -m pip install -r requirements.txt
```

2. Generate ML model files (required before running the backend):
```bash
   cd MLModel
   python ml_v1_tfidf.py
```
   This saves `svm_model_v1.joblib` and `lr_model_v1.joblib` to the `backend/` folder.

3. Start the backend:
```bash
   cd backend
   python -m fastapi dev main.py
```

4. Build the frontend:
```bash
   cd frontend
   node build.mjs
```
