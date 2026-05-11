"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Prediction = {
  label: string;
  score: number;
};

const API_URL = process.env.NEXT_PUBLIC_DOODLE_API_URL ?? "http://127.0.0.1:8000/predict";
const CANVAS_SIZE = 480;
const MODEL_SIZE = 96;
const SMOOTHING_WINDOW = 6;

export function DoodleCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const isDrawingRef = useRef(false);
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);
  const predictTimeoutRef = useRef<number | null>(null);
  const predictionHistoryRef = useRef<Prediction[][]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [status, setStatus] = useState("Draw on the canvas to start live recognition.");
  const [isPredicting, setIsPredicting] = useState(false);

  const sortedPredictions = useMemo(
    () => [...predictions].sort((a, b) => b.score - a.score).slice(0, 5),
    [predictions]
  );

  const setupCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.strokeStyle = "#111111";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = 16;
  }, []);

  useEffect(() => {
    setupCanvas();
  }, [setupCanvas]);

  const getPoint = (event: PointerEvent | React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return { x: 0, y: 0 };
    }
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * CANVAS_SIZE,
      y: ((event.clientY - rect.top) / rect.height) * CANVAS_SIZE
    };
  };

  const drawSegment = (x: number, y: number) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx) {
      return;
    }

    const lastPoint = lastPointRef.current;
    if (!lastPoint) {
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = "#111111";
      ctx.fill();
    } else {
      ctx.beginPath();
      ctx.moveTo(lastPoint.x, lastPoint.y);
      ctx.lineTo(x, y);
      ctx.stroke();
    }

    lastPointRef.current = { x, y };
  };

  const canvasToDataUrl = () => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return "";
    }

    const offscreen = document.createElement("canvas");
    offscreen.width = MODEL_SIZE;
    offscreen.height = MODEL_SIZE;
    const ctx = offscreen.getContext("2d");
    if (!ctx) {
      return "";
    }

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
    ctx.drawImage(canvas, 0, 0, MODEL_SIZE, MODEL_SIZE);
    return offscreen.toDataURL("image/png");
  };

  const requestPrediction = useCallback(async () => {
    const image = canvasToDataUrl();
    if (!image) {
      return;
    }

    setIsPredicting(true);
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image, topk: 5 })
      });

      if (!response.ok) {
        throw new Error(`API returned ${response.status}`);
      }

      const data = (await response.json()) as { predictions: Prediction[] };
      predictionHistoryRef.current.push(data.predictions);
      if (predictionHistoryRef.current.length > SMOOTHING_WINDOW) {
        predictionHistoryRef.current.shift();
      }

      const scoreMap = new Map<string, number>();
      for (const frame of predictionHistoryRef.current) {
        for (const item of frame) {
          scoreMap.set(item.label, (scoreMap.get(item.label) ?? 0) + item.score);
        }
      }
      const divisor = predictionHistoryRef.current.length;
      const smoothed = Array.from(scoreMap.entries()).map(([label, score]) => ({
        label,
        score: score / divisor
      }));

      setPredictions(smoothed);
      setStatus("Live predictions are updating as you draw.");
    } catch {
      setStatus("Could not reach the prediction API. Start the Python server first.");
    } finally {
      setIsPredicting(false);
    }
  }, []);

  const schedulePrediction = useCallback(() => {
    if (predictTimeoutRef.current) {
      window.clearTimeout(predictTimeoutRef.current);
    }
    predictTimeoutRef.current = window.setTimeout(() => {
      void requestPrediction();
    }, 350);
  }, [requestPrediction]);

  const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    isDrawingRef.current = true;
    lastPointRef.current = null;
    const point = getPoint(event);
    drawSegment(point.x, point.y);
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current) {
      return;
    }
    const point = getPoint(event);
    drawSegment(point.x, point.y);
    schedulePrediction();
  };

  const handlePointerUp = () => {
    isDrawingRef.current = false;
    lastPointRef.current = null;
    schedulePrediction();
  };

  const clearCanvas = () => {
    setupCanvas();
    predictionHistoryRef.current = [];
    setPredictions([]);
    setStatus("Canvas cleared. Draw a new object.");
  };

  return (
    <section className="playground">
      <motion.div
        className="canvas-card"
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3, duration: 0.6 }}
      >
        <div className="card-header">
          <div>
            <h2>Sketch board</h2>
            <p>{status}</p>
          </div>
          <button className="ghost-button" onClick={clearCanvas} type="button">
            Clear
          </button>
        </div>

        <canvas
          ref={canvasRef}
          width={CANVAS_SIZE}
          height={CANVAS_SIZE}
          className="drawing-surface"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerLeave={handlePointerUp}
        />
      </motion.div>

      <motion.aside
        className="prediction-card"
        initial={{ opacity: 0, x: 24 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.4, duration: 0.6 }}
      >
        <div className="card-header">
          <div>
            <h2>Model guesses</h2>
            <p>{isPredicting ? "Thinking..." : "Top predictions from the classifier"}</p>
          </div>
          <span className="status-pill">{isPredicting ? "Live" : "Ready"}</span>
        </div>

        <div className="prediction-list">
          <AnimatePresence mode="popLayout">
            {sortedPredictions.length === 0 ? (
              <motion.div
                className="empty-state"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                Start drawing to see live guesses.
              </motion.div>
            ) : (
              sortedPredictions.map((item, index) => (
                <motion.div
                  key={item.label}
                  className="prediction-row"
                  initial={{ opacity: 0, y: 14 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -14 }}
                  transition={{ delay: index * 0.05, duration: 0.3 }}
                >
                  <div>
                    <span className="prediction-rank">#{index + 1}</span>
                    <h3>{item.label}</h3>
                  </div>
                  <div className="score-wrap">
                    <span>{(item.score * 100).toFixed(1)}%</span>
                    <div className="score-bar">
                      <motion.div
                        className="score-fill"
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.max(6, item.score * 100)}%` }}
                        transition={{ duration: 0.35 }}
                      />
                    </div>
                  </div>
                </motion.div>
              ))
            )}
          </AnimatePresence>
        </div>
      </motion.aside>
    </section>
  );
}
