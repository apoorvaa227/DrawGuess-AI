"use client";

import { motion } from "framer-motion";
import { DoodleCanvas } from "../components/DoodleCanvas";

export default function Home() {
  return (
    <main className="page-shell">
      <motion.div
        className="hero-glow"
        animate={{ scale: [1, 1.15, 1], opacity: [0.3, 0.5, 0.3] }}
        transition={{ duration: 8, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut" }}
      />

      <section className="hero-copy">
        <motion.span
          className="eyebrow"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          Live Doodle Recognition
        </motion.span>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.6 }}
        >
          Draw anything. Watch the model guess it in real time.
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.6 }}
        >
          Built with Next.js on the frontend and your trained PyTorch classifier on the backend.
          The UI updates continuously while the user sketches, just like a scribble game.
        </motion.p>
      </section>

      <DoodleCanvas />
    </main>
  );
}
