## Skribbl extension (local predictor)

This extension captures the drawing canvas from `skribbl.io` and sends it to your local FastAPI model at `http://127.0.0.1:8000/predict`.

### 1) Start the model API

From `C:\Users\apoor\Downloads\archive (1)`:

```powershell
python -m uvicorn serve_doodle_api:app --host 127.0.0.1 --port 8000 --reload
```

Optional: verify it is running by opening `http://127.0.0.1:8000/health`.

### 2) Load the extension (Chrome/Edge)

1. Open Extensions page:
   - Chrome: `chrome://extensions`
   - Edge: `edge://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select this folder:
   - `C:\Users\apoor\Downloads\archive (1)\extension`

### 3) Use it

1. Open a `skribbl.io` game tab.
2. Click the extension icon.
3. Click **Capture**
4. Click **Predict**

If the popup says it cannot find the canvas, refresh the skribbl tab and try again.

