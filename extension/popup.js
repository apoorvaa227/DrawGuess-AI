const API_URL = "http://127.0.0.1:8000/predict";
const WORDLIST_CSV = chrome.runtime.getURL("Skribbl-words.csv");

const statusEl = document.getElementById("status");
const previewImg = document.getElementById("previewImg");
const captureBtn = document.getElementById("captureBtn");
const predictBtn = document.getElementById("predictBtn");
const resultsList = document.getElementById("resultsList");
const apiUrlEl = document.getElementById("apiUrl");
const hintValueEl = document.getElementById("hintValue");

apiUrlEl.textContent = API_URL;

let lastCaptureDataUrl = "";
let lastHint = "";
let wordlistCache = null; // Array<{word: string, count: number}>
let wordlistMaxCount = 1;

function setStatus(text) {
  statusEl.textContent = text;
}

function clearResults() {
  resultsList.innerHTML = "";
}

function renderResults(predictions) {
  clearResults();
  for (const item of predictions) {
    const li = document.createElement("li");

    const left = document.createElement("span");
    left.className = "label";
    left.textContent = item.label;

    const right = document.createElement("span");
    right.className = "score";
    right.textContent = `${(item.score * 100).toFixed(1)}%`;

    li.appendChild(left);
    li.appendChild(right);
    resultsList.appendChild(li);
  }
}

async function loadWordlist() {
  if (wordlistCache) return wordlistCache;

  const res = await fetch(WORDLIST_CSV);
  if (!res.ok) throw new Error("Could not load Skribbl word list.");
  const text = await res.text();

  const lines = text.split(/\r?\n/).filter(Boolean);
  // Expected header: word,count,
  const items = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    const parts = line.split(",");
    const word = (parts[0] || "").trim();
    const count = Number.parseInt((parts[1] || "0").trim(), 10) || 0;
    if (!word) continue;
    items.push({ word, count });
    if (count > wordlistMaxCount) wordlistMaxCount = count;
  }

  // Most frequent first
  items.sort((a, b) => b.count - a.count);
  wordlistCache = items;
  return items;
}

async function getActiveTabId() {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  return tabs[0]?.id ?? null;
}

async function captureFromContentScript() {
  const tabId = await getActiveTabId();
  if (!tabId) {
    throw new Error("No active tab found.");
  }

  const response = await chrome.tabs.sendMessage(tabId, { type: "CAPTURE_CANVAS" });
  if (!response?.ok) {
    throw new Error(response?.error ?? "Capture failed.");
  }
  return { dataUrl: response.dataUrl, hint: response.hint ?? "" };
}

function normalizeLabel(label) {
  return String(label || "")
    .toLowerCase()
    .replace(/[-_]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function buildRegexFromHint(hint) {
  // Example hint: "_ a _ _ _" or "_ _ _ _ _  _ _" etc.
  // We treat underscores as wildcards [a-z] and keep known letters fixed.
  // Spaces represent spaces between words (or letter spacing). We'll compact letter spacing.
  const raw = String(hint || "").trim();
  if (!raw || !raw.includes("_")) return null;

  // Remove duplicate spacing; keep word boundaries.
  const tokens = raw.split(/\s+/);
  // skribbl typically spaces between every character; join tokens into words separated by double space markers.
  // We'll rebuild by detecting if tokens contain single characters or underscores.
  // Strategy: interpret tokens as character cells; rebuild a pattern string without spaces between chars.
  let pattern = "";
  for (const t of tokens) {
    if (t === "|") {
      pattern += " ";
      continue;
    }
    // If token itself contains multiple underscores/letters (rare), keep it.
    pattern += t;
  }

  // If the pattern has no spaces but original had obvious word breaks (multiple spaces), we can't recover perfectly.
  // Still, matching without spaces is better than nothing. We'll also try a space-insensitive match later.

  // Escape regex special chars except underscore and letters/spaces.
  const escaped = pattern.replace(/[-/\\^$*+?.()|[\]{}]/g, "\\$&");

  // Convert underscores to letter wildcards. Allow any letter for unknown positions.
  const regexBody = escaped.replace(/_/g, "[a-z]");

  return new RegExp(`^${regexBody}$`, "i");
}

function filterPredictionsByHint(predictions, hint) {
  const rx = buildRegexFromHint(hint);
  if (!rx) return predictions;

  const filtered = predictions.filter((p) => rx.test(normalizeLabel(p.label)));
  return filtered.length > 0 ? filtered : predictions;
}

function rankWordlistWithModel(wordlist, modelPreds, hint) {
  const rx = buildRegexFromHint(hint);
  const modelMap = new Map();
  for (const p of modelPreds) {
    modelMap.set(normalizeLabel(p.label), p.score);
  }

  const filteredWords = rx
    ? wordlist.filter((w) => rx.test(normalizeLabel(w.word)))
    : wordlist.slice();

  // Combine: model confidence (if any) + frequency prior.
  // If the word isn't in model labels, it can still win due to hint+frequency.
  const alpha = 0.75; // weight for model if available
  const scored = filteredWords.map((w) => {
    const key = normalizeLabel(w.word);
    const modelScore = modelMap.get(key) ?? 0;
    const freqScore = w.count / Math.max(1, wordlistMaxCount);
    const score = alpha * modelScore + (1 - alpha) * freqScore;
    return { label: w.word, score };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored;
}

async function predict(dataUrl) {
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataUrl, topk: 5 })
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}`);
  }
  return await res.json();
}

captureBtn.addEventListener("click", async () => {
  setStatus("Capturing from skribbl canvas...");
  clearResults();
  predictBtn.disabled = true;

  try {
    const { dataUrl, hint } = await captureFromContentScript();
    lastCaptureDataUrl = dataUrl;
    lastHint = hint;
    hintValueEl.textContent = hint ? hint : "—";
    previewImg.src = dataUrl;
    predictBtn.disabled = false;
    setStatus("Captured. Click Predict to run the model.");
  } catch (e) {
    setStatus(`Capture error: ${e.message}`);
  }
});

predictBtn.addEventListener("click", async () => {
  if (!lastCaptureDataUrl) return;
  setStatus("Calling FastAPI model...");
  clearResults();

  try {
    const data = await predict(lastCaptureDataUrl);
    const preds = data.predictions ?? [];
    // If we have a hint, use the skribbl wordlist for better matching.
    if (lastHint && lastHint.includes("_")) {
      setStatus("Ranking skribbl wordlist using hint + model...");
      const wordlist = await loadWordlist();
      const ranked = rankWordlistWithModel(wordlist, preds, lastHint).slice(0, 15);
      renderResults(ranked);
    } else {
      const filtered = filterPredictionsByHint(preds, lastHint);
      renderResults(filtered);
    }
    setStatus("Done.");
  } catch (e) {
    setStatus(`Predict error: ${e.message}`);
  }
});

