import { useEffect, useRef, useState } from 'react';
import { jsPDF } from 'jspdf';

const EMPTY_SUMMARY = {
  engine: 'PP-StructureV3',
  page_count: 0,
  table_count: 0,
  text_characters: 0,
};

const OCR_MODES = [
  {
    value: 'fast',
    label: 'Fast',
    description: 'Safer on older Macs. Uses smaller models, downscales oversized images, and skips table recognition.',
  },
  {
    value: 'full',
    label: 'Full',
    description: 'More accurate but heavier. Uses larger models and keeps table recognition on.',
  },
];

function createEmptyResult() {
  return {
    summary: EMPTY_SUMMARY,
    tables: [],
    raw_text: '',
    processing: null,
    svm_prediction: null,
    lr_prediction: null,
  };
}

export default function App() {
  const copyTimerRef = useRef(null);
  const runStartRef = useRef(null);
  const reportUrlRef = useRef('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [badgeStates, setBadgeStates] = useState([]);
  const [activePreviewIndex, setActivePreviewIndex] = useState(0);
  const [processingPreviewIndex, setProcessingPreviewIndex] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [previewUrls, setPreviewUrls] = useState([]);
  const [ocrMode, setOcrMode] = useState('full');
  const [elapsedMs, setElapsedMs] = useState(0);
  const [lastRunMs, setLastRunMs] = useState(0);
  const [status, setStatus] = useState({ message: '', isError: false, showSpinner: false });
  const [result, setResult] = useState(createEmptyResult);
  const [copiedKey, setCopiedKey] = useState('');
  const [reportState, setReportState] = useState({ message: '', isError: false, isGenerating: false });

  const fileMeta = getFileMeta(selectedFiles);
  const currentMode = OCR_MODES.find((mode) => mode.value === ocrMode) || OCR_MODES[0];
  const summary = result.summary || EMPTY_SUMMARY;
  const reportDocuments = getDocumentEntries(result);
  const displayTables = dedupeTables(result.tables);
  const canOpenReport = !isRunning && selectedFiles.length > 0 && reportDocuments.length > 0;
  const highlightedPreviewIndex = processingPreviewIndex ?? activePreviewIndex;

  useEffect(() => {
    if (!selectedFiles.length) {
      setPreviewUrls([]);
      return undefined;
    }

    const urls = selectedFiles.map((f) => URL.createObjectURL(f));
    setPreviewUrls(urls);

    return () => {
      urls.forEach((u) => URL.revokeObjectURL(u));
    };
  }, [selectedFiles]);

  useEffect(() => {
    return () => {
      if (copyTimerRef.current) {
        clearTimeout(copyTimerRef.current);
      }

      if (reportUrlRef.current) {
        URL.revokeObjectURL(reportUrlRef.current);
        reportUrlRef.current = '';
      }
    };
  }, []);

  useEffect(() => {
    if (!isRunning || !runStartRef.current) {
      return undefined;
    }

    const updateElapsed = () => {
      setElapsedMs(Date.now() - runStartRef.current);
    };

    updateElapsed();
    const intervalId = window.setInterval(updateElapsed, 1000);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [isRunning]);

  function resetRunArtifacts() {
    setElapsedMs(0);
    setLastRunMs(0);
    setResult(createEmptyResult());
    setStatus({ message: '', isError: false, showSpinner: false });
    setReportState({ message: '', isError: false, isGenerating: false });

    if (reportUrlRef.current) {
      URL.revokeObjectURL(reportUrlRef.current);
      reportUrlRef.current = '';
    }
  }

  function handleFiles(rawFiles) {
    if (isRunning) {
      return;
    }

    const files = Array.from(rawFiles || []).filter(isSupportedFile);
    if (!files.length) {
      setStatus({ message: 'Upload a PDF or image file.', isError: true, showSpinner: false });
      return;
    }

    setSelectedFiles(files);
    setBadgeStates(files.map(() => 'queued'));
    setActivePreviewIndex(0);
    setProcessingPreviewIndex(null);
    resetRunArtifacts();
  }

  function handleRemoveFile(indexToRemove) {
    if (isRunning) {
      return;
    }

    const nextFiles = selectedFiles.filter((_, index) => index !== indexToRemove);
    setSelectedFiles(nextFiles);
    setBadgeStates(nextFiles.map(() => 'queued'));
    setActivePreviewIndex((currentIndex) => {
      if (!nextFiles.length) return 0;
      if (currentIndex > indexToRemove) return currentIndex - 1;
      return Math.min(currentIndex, nextFiles.length - 1);
    });
    setProcessingPreviewIndex(null);
    resetRunArtifacts();
  }

  function handleInputChange(event) {
    handleFiles(event.target.files);
    event.target.value = '';
  }

  function handleDragOver(event) {
    event.preventDefault();
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = 'copy';
    }
    setDragging(true);
  }

  function handleDragLeave(event) {
    if (!event.currentTarget.contains(event.relatedTarget)) {
      setDragging(false);
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    setDragging(false);

    if (isRunning) return;
    handleFiles(getDroppedFiles(event.dataTransfer));
  }

  function handleFileKeyDown(event, index) {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    setActivePreviewIndex(index);
  }

  async function handleRun() {
    if (!selectedFiles.length || isRunning) return;

    const filesToProcess = selectedFiles;
    const allResults = [];
    let errorCount = 0;

    runStartRef.current = Date.now();
    setIsRunning(true);
    setElapsedMs(0);
    setLastRunMs(0);
    setProcessingPreviewIndex(0);
    setBadgeStates(filesToProcess.map(() => 'queued'));
    setReportState({ message: '', isError: false, isGenerating: false });

    for (let index = 0; index < filesToProcess.length; index += 1) {
      const file = filesToProcess[index];
      setProcessingPreviewIndex(index);
      setBadgeStates((current) => current.map((state, i) => (i === index ? 'running' : state)));
      setStatus({
        message: filesToProcess.length > 1
          ? `Processing ${index + 1} of ${filesToProcess.length}: ${file.name}...`
          : 'Running PP-StructureV3. The first run may take several minutes while models download.',
        isError: false,
        showSpinner: true,
      });

      const formData = new FormData();
      formData.append('file', file);
      formData.append('ocr_mode', ocrMode);

      try {
        const response = await fetch('/ocr', { method: 'POST', body: formData });
        if (!response.ok) {
          const errorPayload = await response.json().catch(() => ({ detail: response.statusText }));
          throw new Error(errorPayload.detail || response.statusText);
        }

        const data = normalizeRunResult(await response.json());
        allResults.push(data);
        setBadgeStates((current) => current.map((state, i) => (i === index ? 'done' : state)));
      } catch (error) {
        errorCount += 1;
        setBadgeStates((current) => current.map((state, i) => (i === index ? 'error' : state)));
        setStatus({ message: `Error on ${file.name}: ${error.message}`, isError: true, showSpinner: false });
      }
    }

    const runDuration = runStartRef.current ? Date.now() - runStartRef.current : 0;
    runStartRef.current = null;
    setElapsedMs(runDuration);
    setLastRunMs(runDuration);
    setIsRunning(false);
    setProcessingPreviewIndex(null);

    if (!allResults.length) return;

    setResult(mergeResults(allResults));
    setStatus({
      message: errorCount > 0
        ? `Done. ${allResults.length} succeeded, ${errorCount} failed.`
        : filesToProcess.length > 1
          ? `All ${filesToProcess.length} files processed.`
          : 'Extraction complete.',
      isError: errorCount > 0,
      showSpinner: false,
    });
  }

  async function copyText(key, content) {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedKey(key);
      if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
      copyTimerRef.current = window.setTimeout(() => setCopiedKey(''), 1500);
    } catch {
      // Clipboard access is optional in this demo.
    }
  }

  async function handleOpenReport() {
    if (!canOpenReport || reportState.isGenerating) return;

    const previewTab = window.open('', '_blank');
    if (previewTab) {
      previewTab.document.title = 'Preparing OCR report';
      previewTab.document.body.innerHTML = '<div style="font-family: Avenir Next, Segoe UI, sans-serif; padding: 32px; color: #16223b;">Preparing PDF report...</div>';
    }

    setReportState({ message: '', isError: false, isGenerating: true });

    try {
      const pdfBlob = await buildReportPdf({
        documents: reportDocuments,
        files: matchFilesToDocuments(selectedFiles, reportDocuments),
        summary,
        lastRunMs,
        modeLabel: result.processing?.ocr_mode_label || currentMode.label,
      });

      if (reportUrlRef.current) URL.revokeObjectURL(reportUrlRef.current);

      const reportUrl = URL.createObjectURL(pdfBlob);
      reportUrlRef.current = reportUrl;

      if (previewTab) {
        previewTab.location.replace(reportUrl);
      } else if (!window.open(reportUrl, '_blank')) {
        throw new Error('Allow pop-ups to open the PDF report.');
      }

      setReportState({ message: 'Report PDF opened in a new tab.', isError: false, isGenerating: false });
    } catch (error) {
      if (previewTab && !previewTab.closed) previewTab.close();
      setReportState({
        message: error?.message || 'Unable to build the PDF report.',
        isError: true,
        isGenerating: false,
      });
    }
  }

  return (
    <main className="page-shell">
      <section className="hero">
        <p className="eyebrow">OCR Demo</p>
        <h1>Pipeline 4 Structured Extraction</h1>
        <p className="lede">
          Upload a screenshot or PDF and inspect the structured output produced by
          {' '}
          <strong>PP-StructureV3</strong>
          .
        </p>
      </section>

      <section className="top-grid">
        <div className="panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">Input</p>
              <h2>Upload Document</h2>
            </div>
            <button type="button" onClick={handleRun} disabled={!selectedFiles.length || isRunning}>
              {isRunning ? 'Running...' : 'Run P4'}
            </button>
          </div>

          <div
            className={`upload-area${dragging ? ' dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              id="file-input"
              type="file"
              accept="image/*,.pdf,application/pdf"
              multiple
              disabled={isRunning}
              onChange={handleInputChange}
            />
            <label htmlFor="file-input">
              Drop images or PDFs here
              <span>or browse your files — multiple files supported</span>
            </label>
          </div>

          <div className="mode-strip">
            <div className="mode-copy">
              <p className="mode-label">OCR Mode</p>
              <p className="mode-description">{currentMode.description}</p>
            </div>
            <div className="mode-toggle" role="tablist" aria-label="OCR mode">
              {OCR_MODES.map((mode) => (
                <button
                  key={mode.value}
                  type="button"
                  className={`mode-toggle-btn${ocrMode === mode.value ? ' active' : ''}`}
                  onClick={() => setOcrMode(mode.value)}
                  disabled={isRunning}
                >
                  {mode.label}
                </button>
              ))}
            </div>
          </div>

          <div className="file-meta">{fileMeta}</div>
          {selectedFiles.length > 0 ? (
            <div className="file-queue">
              {selectedFiles.length > 1 && (
                <p className="file-queue-label">
                  Queue
                  <span>{selectedFiles.length} files</span>
                </p>
              )}
              <ul className="file-list">
                {selectedFiles.map((file, index) => {
                  const state = badgeStates[index] || 'queued';
                  return (
                    <li
                      key={`${file.name}-${file.size}-${index}`}
                      className={`file-list-item${index === highlightedPreviewIndex ? ' active' : ''}`}
                      tabIndex={0}
                      role="button"
                      aria-pressed={index === highlightedPreviewIndex}
                      onClick={() => setActivePreviewIndex(index)}
                      onKeyDown={(event) => handleFileKeyDown(event, index)}
                    >
                      <span className="file-list-num">{String(index + 1).padStart(2, '0')}</span>
                      <FileIcon file={file} />
                      <span className="file-list-name">{file.name}</span>
                      <span className="file-list-size">{formatBytes(file.size)}</span>
                      <span className="file-list-status" data-state={state}>
                        <span className="file-list-dot" />
                        <span className="file-list-status-label">{state}</span>
                      </span>
                    </li>
                  );
                })}
              </ul>
            </div>
          ) : null}

          <div className="preview-shell">
            {!selectedFiles.length ? (
              <div className="empty-preview">Preview will appear here.</div>
            ) : (
              <div className="thumb-grid">
                {selectedFiles.map((file, index) => (
                  <div
                    key={`${file.name}-${index}`}
                    className={`thumb-cell${index === highlightedPreviewIndex ? ' active' : ''}`}
                  >
                    <button
                      type="button"
                      className="thumb-remove"
                      aria-label={`Remove ${file.name}`}
                      title={`Remove ${file.name}`}
                      disabled={isRunning}
                      onClick={() => handleRemoveFile(index)}
                    >
                      ×
                    </button>
                    <button
                      type="button"
                      className="thumb-card"
                      onClick={() => setActivePreviewIndex(index)}
                      title={file.name}
                    >
                      {isPdfFile(file) ? (
                        <div className="thumb-pdf-placeholder">PDF</div>
                      ) : (
                        <img
                          className="thumb-img"
                          src={previewUrls[index] || ''}
                          alt={file.name}
                        />
                      )}
                      <span className="thumb-label">{file.name}</span>
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="status-row">
            <div className={status.isError ? 'status error' : 'status'}>
              <span className={`spinner${status.showSpinner ? ' active' : ''}`} />
              <span>{status.message}</span>
            </div>
            {(isRunning || lastRunMs > 0) ? (
              <div className="timer-chip">
                <span className="timer-chip-label">{isRunning ? 'Elapsed' : 'Last run'}</span>
                <strong>{formatDuration(isRunning ? elapsedMs : lastRunMs)}</strong>
              </div>
            ) : null}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">Snapshot</p>
              <h2>Extraction Summary</h2>
            </div>
          </div>
          <div className="summary-grid">
            <StatCard label="Engine" value={summary.engine || 'PP-StructureV3'} />
            <StatCard label="Pages" value={summary.page_count ?? 0} />
            <StatCard label="Tables" value={summary.table_count ?? 0} />
            <StatCard label="Text Chars" value={summary.text_characters ?? 0} />
          </div>
          <p className="summary-note">
            The backend returns raw OCR text, detected tables, and ML classification results.
          </p>
          <div className="summary-actions">
            <button
              type="button"
              className="summary-report-btn"
              onClick={handleOpenReport}
              disabled={!canOpenReport || reportState.isGenerating}
            >
              {reportState.isGenerating ? 'Preparing PDF...' : 'Open Report PDF'}
            </button>
            <p className={reportState.isError ? 'summary-feedback error' : 'summary-feedback'}>
              {reportState.message || 'Creates a readable PDF with each source image and its OCR output.'}
            </p>
          </div>
        </div>
      </section>

      <section className="results-grid">

        {result.svm_prediction && (
          <PredictionCard
            prediction={result.svm_prediction}
            title="SVM Classification"
            kicker="SVM · CV mean 100% ± 0%"
          />
        )}

        {result.lr_prediction && (
          <PredictionCard
            prediction={result.lr_prediction}
            title="Logistic Regression Classification"
            kicker="LR · CV mean 91% ± 8.6%"
          />
        )}

        <div className="panel tables-panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">Structured Regions</p>
              <h2>Detected Tables</h2>
            </div>
          </div>
          <div className="tables-stack">
            {displayTables.length > 0 ? displayTables.map((table, index) => (
              <article key={`table-${index}`} className="mini-table-card">
                <div className="mini-table-head">
                  <strong>{`Table ${index + 1}`}</strong>
                  <span>{`BBox ${Array.isArray(table.bbox) ? table.bbox.join(', ') : '0, 0, 0, 0'}`}</span>
                </div>
                <div className="table-shell">
                  {(table.rows || []).length > 0 ? (
                    <table className="results-table compact generic-table">
                      <tbody>
                        {(table.rows || []).map((row, rowIndex) => (
                          <tr key={`row-${rowIndex}`}>
                            {row.map((cell, cellIndex) => (
                              <td key={`cell-${rowIndex}-${cellIndex}`}>{cell || ''}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <table className="results-table compact">
                      <tbody>
                        <tr className="empty-row">
                          <td colSpan="2">No parsed cells.</td>
                        </tr>
                      </tbody>
                    </table>
                  )}
                </div>
              </article>
            )) : (
              <p className="empty-copy">
                {result.processing?.ocr_mode === 'fast'
                  ? 'Fast mode skips table recognition to reduce load.'
                  : 'Detected tables will appear here.'}
              </p>
            )}
          </div>
        </div>

        <div className="panel markdown-panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">Text Output</p>
              <h2>OCR Text</h2>
            </div>
            <button
              type="button"
              className="copy-btn"
              onClick={() => copyText('raw_text', result.raw_text || 'No text extracted.')}
            >
              {copiedKey === 'raw_text' ? 'Copied!' : 'Copy'}
            </button>
          </div>
          <pre className="code-block">{result.raw_text || 'No text extracted.'}</pre>
        </div>

        <div className="panel">
          <div className="panel-header">
            <div>
              <p className="section-kicker">Debug View</p>
              <h2>Raw JSON</h2>
            </div>
            <button
              type="button"
              className="copy-btn"
              onClick={() => copyText('json', JSON.stringify({ features: result.features, tfidf: result.tfidf }, null, 2))}
            >
              {copiedKey === 'json' ? 'Copied!' : 'Copy'}
            </button>
          </div>
          <pre className="code-block">
            {result.features
              ? JSON.stringify({ features: result.features, tfidf: result.tfidf }, null, 2)
              : JSON.stringify(result, null, 2)}
          </pre>
        </div>
      </section>
    </main>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="stat-card">
      <span className="stat-label">{label}</span>
      <strong className="stat-value">{value}</strong>
    </div>
  );
}

function PredictionCard({ prediction, title, kicker }) {
  if (!prediction || prediction.label === 'unavailable') return null;

  const isUnknown = prediction.label === 'Unknown';
  const isLowConf = prediction.confidence < 0.6;
  const confPct   = (prediction.confidence * 100).toFixed(1);
  const probs     = Object.entries(prediction.all_probs || {}).sort((a, b) => b[1] - a[1]);
  const topClass  = probs[0]?.[0];

  return (
    <div className="panel pred-card">
      <div className="panel-header">
        <div>
          <p className="section-kicker">{kicker || 'ML Classifier'}</p>
          <h2>{title || 'Document Classification'}</h2>
        </div>
      </div>
      <div className="pred-layout">
        <div className="pred-main">
          <div className={`pred-label${isUnknown ? ' unknown' : ''}`}>
            {isUnknown ? '— Unknown —' : prediction.label}
          </div>
          <div className={`pred-confidence${isLowConf ? ' low' : ''}`}>
            {confPct}% confidence
          </div>
          {isUnknown && <div className="pred-threshold-note">Below 60% threshold</div>}
        </div>
        <div className="pred-divider" />
        <div className="pred-probs">
          {probs.map(([name, prob]) => {
            const pct   = (prob * 100).toFixed(1);
            const isTop = name === topClass;
            return (
              <div key={name} className="pred-prob-row">
                <div className="pred-prob-header">
                  <span className="pred-prob-name">{name}</span>
                  <span className="pred-prob-pct">{pct}%</span>
                </div>
                <div className="pred-prob-bar-bg">
                  <div
                    className={`pred-prob-bar-fill${isTop ? ' top' : ' other'}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Data helpers
// ---------------------------------------------------------------------------

function normalizeRunResult(result) {
  return {
    source:         result.source || '',
    summary:        normalizeSummary(result.summary),
    raw_text:       result.raw_text || '',
    tables:         Array.isArray(result.tables) ? result.tables : [],
    processing:     result.processing || null,
    svm_prediction: result.svm_prediction || null,
    lr_prediction:  result.lr_prediction  || null,
    features:       result.features || null,
    tfidf:          result.tfidf    || null,
  };
}

function normalizeSummary(summary) {
  return { ...EMPTY_SUMMARY, ...(summary || {}) };
}

function mergeResults(results) {
  if (results.length === 1) {
    return { ...results[0], documents: results };
  }

  const textParts = results.map((r) => {
    const source = r.source || '';
    const text   = r.raw_text || '';
    return source ? `### ${source}\n\n${text}` : text;
  });

  return {
    source:         results.map((r) => r.source).join(', '),
    summary: {
      engine:          'PP-StructureV3',
      page_count:      results.reduce((sum, r) => sum + (r.summary?.page_count      ?? 0), 0),
      table_count:     results.reduce((sum, r) => sum + (r.summary?.table_count     ?? 0), 0),
      text_characters: results.reduce((sum, r) => sum + (r.summary?.text_characters ?? 0), 0),
    },
    raw_text:       textParts.join('\n\n---\n\n'),
    tables:         results.flatMap((r) => r.tables || []),
    processing:     results[0]?.processing || null,
    svm_prediction: null,
    lr_prediction:  null,
    documents:      results,
  };
}

function getDocumentEntries(result) {
  if (Array.isArray(result.documents) && result.documents.length) {
    return result.documents;
  }
  return [result];
}

function dedupeTables(tables) {
  const seen = new Set();
  return (tables || []).filter((table) => {
    const rows = Array.isArray(table.rows) ? table.rows : [];
    const signature = JSON.stringify(rows.slice(0, 6));
    if (seen.has(signature)) return false;
    seen.add(signature);
    return rows.length > 0;
  });
}

function matchFilesToDocuments(files, documents) {
  const nameBuckets = new Map();
  const usedIndexes = new Set();

  files.forEach((file, index) => {
    const bucket = nameBuckets.get(file.name) || [];
    bucket.push(index);
    nameBuckets.set(file.name, bucket);
  });

  return documents.map((document) => {
    const bucket = nameBuckets.get(document.source) || [];
    while (bucket.length > 0) {
      const index = bucket.shift();
      if (!usedIndexes.has(index)) {
        usedIndexes.add(index);
        return files[index];
      }
    }
    const fallbackIndex = files.findIndex((_, index) => !usedIndexes.has(index));
    if (fallbackIndex >= 0) {
      usedIndexes.add(fallbackIndex);
      return files[fallbackIndex];
    }
    return null;
  });
}

// ---------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------

function isSupportedFile(file) {
  if (!file) return false;
  const name = file.name.toLowerCase();
  return file.type.startsWith('image/')
    || file.type === 'application/pdf'
    || name.endsWith('.pdf')
    || name.endsWith('.png')
    || name.endsWith('.jpg')
    || name.endsWith('.jpeg')
    || name.endsWith('.tif')
    || name.endsWith('.tiff')
    || name.endsWith('.bmp');
}

function isPdfFile(file) {
  return file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
}

function getDroppedFiles(dataTransfer) {
  if (!dataTransfer) return [];
  if (dataTransfer.items?.length) {
    return Array.from(dataTransfer.items)
      .filter((item) => item.kind === 'file')
      .map((item) => item.getAsFile())
      .filter(Boolean);
  }
  return Array.from(dataTransfer.files || []);
}

function getFileMeta(files) {
  if (!files.length) return 'No file selected.';
  if (files.length === 1) {
    const [file] = files;
    return `${file.name} • ${formatBytes(file.size)}${file.type ? ` • ${file.type}` : ''}`;
  }
  const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
  return `${files.length} files selected • ${formatBytes(totalBytes)} total • select a row to preview`;
}

function FileIcon({ file }) {
  if (isPdfFile(file)) {
    return (
      <svg className="file-list-icon" viewBox="0 0 20 20" fill="none" aria-hidden="true">
        <rect x="3" y="1" width="11" height="15" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
        <path d="M11 1v4h4" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
        <path d="M6 9h5M6 12h3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        <rect x="10" y="12" width="7" height="5" rx="1" fill="var(--accent)" />
        <text x="13.5" y="16.2" fontSize="3.8" fill="white" textAnchor="middle" fontWeight="700">PDF</text>
      </svg>
    );
  }
  return (
    <svg className="file-list-icon" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <rect x="2" y="2" width="16" height="16" rx="2" stroke="currentColor" strokeWidth="1.5" />
      <circle cx="7.5" cy="7.5" r="1.5" stroke="currentColor" strokeWidth="1.2" />
      <path d="M2 13l4-3.5 3 2.5 3-4 6 5" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatDuration(milliseconds) {
  const totalSeconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

// ---------------------------------------------------------------------------
// PDF report
// ---------------------------------------------------------------------------

async function buildReportPdf({ documents, files, summary, lastRunMs, modeLabel }) {
  const pdf = new jsPDF({ orientation: 'portrait', unit: 'pt', format: 'a4', compress: true });
  const pageWidth    = pdf.internal.pageSize.getWidth();
  const pageHeight   = pdf.internal.pageSize.getHeight();
  const margin       = 42;
  const contentWidth = pageWidth - (margin * 2);
  const generatedAt  = new Date().toLocaleString();

  drawReportCover(pdf, { margin, contentWidth, summary, lastRunMs, generatedAt, modeLabel, documentCount: documents.length });

  for (let index = 0; index < documents.length; index += 1) {
    pdf.addPage();
    await drawDocumentPage(pdf, {
      document: documents[index],
      file: files[index] || null,
      index,
      total: documents.length,
      margin,
      contentWidth,
      pageHeight,
    });
  }

  return pdf.output('blob');
}

function drawReportCover(pdf, { margin, contentWidth, summary, lastRunMs, generatedAt, modeLabel, documentCount }) {
  let y = margin + 12;

  pdf.setFillColor(244, 241, 232);
  pdf.roundedRect(margin, y, contentWidth, 136, 22, 22, 'F');

  pdf.setTextColor(15, 107, 111);
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(11);
  pdf.text('OCR DEMO REPORT', margin + 22, y + 24);

  pdf.setTextColor(22, 34, 59);
  pdf.setFontSize(25);
  pdf.text('Structured Extraction Summary', margin + 22, y + 56);

  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(11);
  pdf.setTextColor(89, 101, 125);
  pdf.text(`Generated ${generatedAt}`, margin + 22, y + 82);
  pdf.text(`Mode ${modeLabel}`, margin + 22, y + 100);
  pdf.text(`Files ${documentCount}`, margin + 22, y + 118);
  if (lastRunMs > 0) pdf.text(`Run time ${formatDuration(lastRunMs)}`, margin + 160, y + 100);

  y += 164;

  const stats = [
    ['Engine',     summary.engine || 'PP-StructureV3'],
    ['Pages',      String(summary.page_count      ?? 0)],
    ['Tables',     String(summary.table_count      ?? 0)],
    ['Text Chars', String(summary.text_characters  ?? 0)],
  ];

  stats.forEach(([label, value], index) => {
    const columnWidth = (contentWidth - 16) / 2;
    const cardX = margin + ((index % 2) * (columnWidth + 16));
    const cardY = y + (Math.floor(index / 2) * 88);

    pdf.setFillColor(255, 255, 255);
    pdf.setDrawColor(221, 229, 242);
    pdf.roundedRect(cardX, cardY, columnWidth, 72, 18, 18, 'FD');

    pdf.setTextColor(89, 101, 125);
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(9);
    pdf.text(label.toUpperCase(), cardX + 16, cardY + 22);

    pdf.setTextColor(22, 34, 59);
    pdf.setFontSize(19);
    pdf.text(value, cardX + 16, cardY + 50);
  });
}

async function drawDocumentPage(pdf, { document, file, index, total, margin, contentWidth, pageHeight }) {
  let y = margin;

  pdf.setTextColor(15, 107, 111);
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(10);
  pdf.text(`SOURCE ${index + 1} OF ${total}`, margin, y);

  y += 22;
  pdf.setTextColor(22, 34, 59);
  pdf.setFontSize(20);
  pdf.text(document.source || `Document ${index + 1}`, margin, y);

  y += 18;
  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(10.5);
  pdf.setTextColor(89, 101, 125);
  pdf.text(buildDocumentMetaLine(document), margin, y);
  y += 24;

  if (file && !isPdfFile(file)) {
    try {
      const image       = await prepareImageForPdf(file);
      const imageBoxH   = 248;
      const imageScale  = Math.min(contentWidth / image.width, imageBoxH / image.height, 1);
      const drawWidth   = image.width  * imageScale;
      const drawHeight  = image.height * imageScale;
      const drawX       = margin + ((contentWidth - drawWidth) / 2);

      pdf.setFillColor(255, 255, 255);
      pdf.setDrawColor(221, 229, 242);
      pdf.roundedRect(margin, y, contentWidth, drawHeight + 24, 18, 18, 'FD');
      pdf.addImage(image.dataUrl, image.format, drawX, y + 12, drawWidth, drawHeight, undefined, 'FAST');
      y += drawHeight + 40;
    } catch {
      y = drawReportPlaceholder(pdf, {
        x: margin, y, width: contentWidth, height: 116,
        title: 'Image preview unavailable',
        subtitle: 'The OCR summary below is still included in this report.',
      });
    }
  } else {
    y = drawReportPlaceholder(pdf, {
      x: margin, y, width: contentWidth, height: 116,
      title: file ? 'PDF source attached to OCR run' : 'Source preview unavailable',
      subtitle: file
        ? `${file.name} • preview omitted from in-browser report export`
        : 'Only OCR text is available for this page.',
    });
  }

  const sections = buildReadableSections(document);
  sections.forEach((section) => {
    y = ensurePdfRoom(pdf, y, 18, { pageHeight, margin });
    pdf.setTextColor(22, 34, 59);
    pdf.setFont('helvetica', 'bold');
    pdf.setFontSize(12.5);
    pdf.text(section.title, margin, y);
    y += 18;

    if (section.type === 'table') {
      y = drawPdfTable(pdf, {
        title: section.title,
        rows: section.rows,
        headerRowCount: section.headerRowCount || 0,
        x: margin, y, width: contentWidth, pageHeight, margin,
      });
      y += 8;
      return;
    }

    pdf.setFont('helvetica', 'normal');
    pdf.setFontSize(10.5);
    pdf.setTextColor(42, 52, 73);
    y = drawWrappedParagraphs(pdf, section.lines, { x: margin, y, maxWidth: contentWidth, pageHeight, margin });
    y += 6;
  });
}

function buildReadableSections(document) {
  const sections      = [];
  const tableSections = getTableSections(document.tables);
  const textLines     = getReadableTextLines(document);

  tableSections.forEach((section) => sections.push(section));

  if (textLines.length > 0) {
    sections.push({ title: 'Readable OCR', lines: textLines });
  }

  if (!sections.length) {
    sections.push({
      title: 'Readable OCR',
      lines: ['No text was extracted for this source. The original screenshot above is included for context.'],
    });
  }

  return sections;
}

function getTableSections(tables) {
  return dedupeTables(tables).slice(0, 3).map((table, index) => {
    const rows = (table.rows || [])
      .map((row) => row.map((cell) => normalizeWhitespace(cell)))
      .filter((row) => row.some(Boolean))
      .slice(0, 10);

    if (rows.length > 10) rows.push([`${(table.rows || []).length - 10} more row(s) not shown`]);

    return { title: `Table ${index + 1}`, type: 'table', headerRowCount: 0, rows };
  }).filter((section) => section.rows.length > 0);
}

function getReadableTextLines(document) {
  const source = document.raw_text || '';
  const seen   = new Set();

  return extractReadableLines(source).filter((line) => {
    if (!line || seen.has(line)) return false;
    seen.add(line);
    return true;
  }).slice(0, 18);
}

function extractReadableLines(source) {
  if (!source) return [];

  return source.split(/\n{2,}/).flatMap((block) => {
    const trimmed = block.trim();
    if (!trimmed || trimmed === '---') return [];
    const text = normalizeWhitespace(trimmed.replace(/^#+\s*/, ''));
    return text ? [text] : [];
  });
}

function buildDocumentMetaLine(document) {
  const summary = document.summary || EMPTY_SUMMARY;
  return [
    `Pages ${summary.page_count ?? 0}`,
    `Tables ${summary.table_count ?? 0}`,
    `Text ${summary.text_characters ?? 0} chars`,
  ].join(' • ');
}

function drawReportPlaceholder(pdf, { x, y, width, height, title, subtitle }) {
  pdf.setFillColor(249, 251, 255);
  pdf.setDrawColor(221, 229, 242);
  pdf.roundedRect(x, y, width, height, 18, 18, 'FD');

  pdf.setTextColor(22, 34, 59);
  pdf.setFont('helvetica', 'bold');
  pdf.setFontSize(13);
  pdf.text(title, x + 18, y + 36);

  pdf.setTextColor(89, 101, 125);
  pdf.setFont('helvetica', 'normal');
  pdf.setFontSize(10.5);
  const wrapped = pdf.splitTextToSize(subtitle, width - 36);
  pdf.text(wrapped, x + 18, y + 58);

  return y + height + 18;
}

function ensurePdfRoom(pdf, y, neededHeight, { pageHeight, margin }) {
  if (y + neededHeight <= pageHeight - margin) return y;
  pdf.addPage();
  return margin;
}

function drawWrappedParagraphs(pdf, lines, { x, y, maxWidth, pageHeight, margin }) {
  lines.filter(Boolean).forEach((line) => {
    const wrapped = pdf.splitTextToSize(line, maxWidth);
    wrapped.forEach((segment) => {
      if (y > pageHeight - margin) { pdf.addPage(); y = margin; }
      pdf.text(segment, x, y);
      y += 14;
    });
    y += 6;
  });
  return y;
}

function drawPdfTable(pdf, { title, rows, headerRowCount = 0, x, y, width, pageHeight, margin }) {
  const normalizedRows = (rows || [])
    .map((row) => Array.isArray(row)
      ? row.map((cell) => normalizeWhitespace(cell || ''))
      : [normalizeWhitespace(row || '')])
    .filter((row) => row.some(Boolean));

  if (!normalizedRows.length) return y;

  const columnCount  = Math.max(...normalizedRows.map((row) => row.length), 1);
  const tableRows    = normalizedRows.map((row) => Array.from({ length: columnCount }, (_, i) => row[i] || ''));
  const columnWidths = getPdfColumnWidths(tableRows, width);
  const fontSize     = columnCount >= 8 ? 7 : columnCount >= 6 ? 7.6 : 8.5;
  const lineHeight   = fontSize * 1.34;
  const cellPaddingX = 6;
  const cellPaddingY = 5;

  const wrappedRows  = tableRows.map((row) =>
    row.map((cell, ci) => clampWrappedLines(pdf.splitTextToSize(cell || ' ', Math.max(columnWidths[ci] - cellPaddingX * 2, 16)), 6))
  );
  const headerRows = wrappedRows.slice(0, headerRowCount);
  const bodyRows   = wrappedRows.slice(headerRowCount);

  pdf.setFontSize(fontSize);

  const renderRow = (wrappedCells, rowY, isHeader) => {
    const rowHeight = Math.max(...wrappedCells.map((lines) => Math.max(lines.length, 1))) * lineHeight + cellPaddingY * 2;
    let cursorX = x;
    wrappedCells.forEach((lines, ci) => {
      const cellWidth = columnWidths[ci];
      pdf.setDrawColor(221, 229, 242);
      pdf.setFillColor(isHeader ? 240 : 255, isHeader ? 244 : 255, isHeader ? 252 : 255);
      pdf.rect(cursorX, rowY, cellWidth, rowHeight, 'FD');
      pdf.setTextColor(isHeader ? 89 : 42, isHeader ? 101 : 52, isHeader ? 125 : 73);
      pdf.setFont('helvetica', isHeader ? 'bold' : 'normal');
      pdf.text(lines, cursorX + cellPaddingX, rowY + cellPaddingY + fontSize, { baseline: 'alphabetic' });
      cursorX += cellWidth;
    });
    return rowHeight;
  };

  const renderHeaderRows = () => {
    headerRows.forEach((wrappedCells) => {
      const rowHeight = Math.max(...wrappedCells.map((lines) => Math.max(lines.length, 1))) * lineHeight + cellPaddingY * 2;
      y = ensurePdfRoom(pdf, y, rowHeight, { pageHeight, margin });
      renderRow(wrappedCells, y, true);
      y += rowHeight;
    });
  };

  renderHeaderRows();

  bodyRows.forEach((wrappedCells) => {
    const rowHeight = Math.max(...wrappedCells.map((lines) => Math.max(lines.length, 1))) * lineHeight + cellPaddingY * 2;
    if (y + rowHeight > pageHeight - margin) {
      pdf.addPage();
      y = margin;
      pdf.setTextColor(22, 34, 59);
      pdf.setFont('helvetica', 'bold');
      pdf.setFontSize(12.5);
      pdf.text(`${title} (cont.)`, x, y);
      y += 18;
      renderHeaderRows();
    }
    renderRow(wrappedCells, y, false);
    y += rowHeight;
  });

  return y;
}

function getPdfColumnWidths(rows, totalWidth) {
  const columnCount = Math.max(...rows.map((row) => row.length), 1);
  const weights     = Array.from({ length: columnCount }, () => 1);
  rows.forEach((row) => {
    row.forEach((cell, i) => {
      weights[i] = Math.max(weights[i], Math.min((cell || '').length, 36) || 1);
    });
  });
  const totalWeight = weights.reduce((sum, v) => sum + v, 0);
  const minWidth    = Math.max(46, totalWidth / (columnCount * 1.7));
  let widths        = weights.map((w) => Math.max((w / totalWeight) * totalWidth, minWidth));
  const widthSum    = widths.reduce((sum, v) => sum + v, 0);
  if (widthSum > totalWidth) {
    const scale = totalWidth / widthSum;
    widths = widths.map((w) => w * scale);
  } else if (widthSum < totalWidth) {
    widths[widths.length - 1] += totalWidth - widthSum;
  }
  return widths;
}

function clampWrappedLines(lines, maxLines) {
  if (lines.length <= maxLines) return lines;
  const visible = lines.slice(0, maxLines);
  visible[maxLines - 1] = `${visible[maxLines - 1].replace(/\s+$/, '')}…`;
  return visible;
}

async function prepareImageForPdf(file) {
  const dataUrl  = await readFileAsDataUrl(file);
  const image    = await loadImage(dataUrl);
  const canvas   = document.createElement('canvas');
  const width    = image.naturalWidth || image.width;
  const height   = image.naturalHeight || image.height;
  canvas.width   = width;
  canvas.height  = height;

  const context = canvas.getContext('2d');
  if (!context) throw new Error('Unable to prepare the image for PDF export.');
  context.drawImage(image, 0, 0, width, height);

  const format = getPdfImageFormat(file);
  return {
    dataUrl: format === 'JPEG' ? canvas.toDataURL('image/jpeg', 0.92) : canvas.toDataURL('image/png'),
    width,
    height,
    format,
  };
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader   = new FileReader();
    reader.onload  = () => resolve(reader.result);
    reader.onerror = () => reject(new Error(`Unable to read ${file.name}.`));
    reader.readAsDataURL(file);
  });
}

function loadImage(dataUrl) {
  return new Promise((resolve, reject) => {
    const image    = new Image();
    image.onload   = () => resolve(image);
    image.onerror  = () => reject(new Error('Unable to load the uploaded image for the PDF report.'));
    image.src      = dataUrl;
  });
}

function getPdfImageFormat(file) {
  return file.type === 'image/jpeg' || file.type === 'image/jpg' ? 'JPEG' : 'PNG';
}

function normalizeWhitespace(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}
