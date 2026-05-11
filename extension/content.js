function findSkribblCanvas() {
  // Skribbl uses a game canvas container with one or more canvas layers.
  // We pick the first canvas inside #game-canvas by default.
  // (If skribbl adds multiple layers later, we can improve this by selecting the canvas with most ink.)
  return document.querySelector("#game-canvas canvas");
}

function findHintPattern() {
  // Primary: skribbl hint UI is rendered as hint cells under `.hints`.
  // - `.hint` is usually `_`
  // - `.hint.uncover` contains revealed letter text
  const hintContainers = Array.from(document.querySelectorAll(".hints .container"));
  if (hintContainers.length > 0) {
    const words = hintContainers
      .map((container) => {
        const cells = Array.from(container.querySelectorAll(".hint, .hint.uncover"));
        const chars = cells.map((el) => {
          const t = (el.textContent || "").trim();
          return t ? t[0] : "_";
        });
        const word = chars.join("").replace(/\s+/g, "");
        return word;
      })
      .filter((w) => w && /_/.test(w));

    if (words.length > 0) {
      return words.join(" ");
    }
  }

  // Secondary fallbacks: legacy/alternate layouts.
  const candidates = [
    document.querySelector("#currentWord"),
    document.querySelector(".currentWord"),
    document.querySelector("[id*='current'][id*='word']"),
    document.querySelector("[class*='current'][class*='word']")
  ].filter(Boolean);

  for (const el of candidates) {
    const text = (el.textContent || "").trim();
    if (text && /_/.test(text)) return text;
  }

  // Fallback: scan small number of elements for underscore-heavy text.
  const els = Array.from(document.querySelectorAll("div, span")).slice(0, 400);
  let best = "";
  for (const el of els) {
    const text = (el.textContent || "").trim();
    if (!text) continue;
    const underscoreCount = (text.match(/_/g) || []).length;
    if (underscoreCount >= 2 && underscoreCount > (best.match(/_/g) || []).length) {
      best = text;
    }
  }
  return best || null;
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message || message.type !== "CAPTURE_CANVAS") return;

  try {
    const canvas = findSkribblCanvas();
    if (!canvas) {
      sendResponse({ ok: false, error: "Could not find skribbl canvas (#game-canvas canvas)." });
      return;
    }

    const dataUrl = canvas.toDataURL("image/png");
    const hint = findHintPattern();
    sendResponse({ ok: true, dataUrl, width: canvas.width, height: canvas.height, hint });
  } catch (err) {
    sendResponse({
      ok: false,
      error: err && err.message ? err.message : String(err)
    });
  }

  // Required for async sendResponse in MV3 content scripts.
  return true;
});

