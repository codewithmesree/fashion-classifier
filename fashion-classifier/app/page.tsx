"use client";

import { useCallback, useState } from "react";
import UploadSwatch from "@/components/UploadSwatch";
import GarmentTag from "@/components/GarmentTag";
import PredictionPanel from "@/components/PredictionPanel";
import { CLASS_NAMES } from "@/lib/classes";
import { predict } from "@/lib/model";

type Status = "idle" | "loading" | "ready" | "error";

export default function Home() {
  const [status, setStatus] = useState<Status>("idle");
  const [probs, setProbs] = useState<number[] | null>(null);

  const handleSample = useCallback(async (pixels: Float32Array) => {
    setStatus("loading");
    try {
      const result = await predict(pixels);
      setProbs(result);
      setStatus("ready");
    } catch (err) {
      console.error(err);
      setStatus("error");
    }
  }, []);

  const handleClear = useCallback(() => {
    setProbs(null);
    setStatus("idle");
  }, []);

  const topIndex =
    probs && status === "ready" && Math.max(...probs) > 0
      ? probs.indexOf(Math.max(...probs))
      : -1;

  return (
    <main className="mx-auto flex min-h-screen max-w-5xl flex-col gap-10 px-6 py-12 sm:py-16">
      <header className="mx-auto max-w-2xl text-center">
        <p className="font-tag text-xs uppercase tracking-tagwide text-thread">
          MobileNetV2 · Real Photos
        </p>
        <h1 className="mt-2 font-display text-4xl font-medium leading-tight text-ink sm:text-5xl">
          Garment Scanner
        </h1>
        <p className="mt-3 font-body text-sm text-ink/70 sm:text-base">
          Upload a photo of a piece of clothing. A MobileNet transfer-learning model
          reads it the way a care label reads fabric — and tells you what it sees.
        </p>
      </header>

      <section className="flex flex-col items-center gap-8 lg:flex-row lg:items-start lg:justify-center lg:gap-14">
        {/* Left Column: Image Upload Swatch */}
        <div className="flex w-full max-w-sm flex-col items-center">
          <UploadSwatch
            onSample={handleSample}
            onClear={handleClear}
            disabled={status === "loading"}
          />
        </div>

        {/* Right Column: Garment Tag & Composition Breakdown */}
        <div className="flex w-full max-w-sm flex-col items-center gap-6">
          <GarmentTag
            label={topIndex >= 0 ? CLASS_NAMES[topIndex] : null}
            confidence={topIndex >= 0 ? probs![topIndex] : null}
          />

          {status === "loading" && (
            <div className="flex items-center gap-2 font-tag text-xs uppercase tracking-tagwide text-muted">
              <span className="inline-block h-2 w-2 animate-ping rounded-full bg-thread" />
              Reading garment…
            </div>
          )}

          {status === "error" && (
            <p className="max-w-xs text-center font-tag text-xs uppercase tracking-tagwide text-thread">
              Inference error. Please ensure public/model/model.json is present.
            </p>
          )}

          <PredictionPanel probabilities={probs} />
        </div>
      </section>
    </main>
  );
}
