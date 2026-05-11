FROM python:3.11-slim

WORKDIR /app

# System deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what we need to serve the API.
COPY serve_doodle_api.py .
COPY runs/doodle_classifier/labels.json runs/doodle_classifier/labels.json

EXPOSE 8000

# MODEL_URL should point to your GitHub Release asset.
# Example:
# docker run -e MODEL_URL="https://github.com/.../releases/download/v1.0.0/best_model.pt" -p 8000:8000 yourimage
CMD ["python", "-m", "uvicorn", "serve_doodle_api:app", "--host", "0.0.0.0", "--port", "8000"]

