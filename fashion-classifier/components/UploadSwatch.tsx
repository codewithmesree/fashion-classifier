"use client";

import { useCallback, useRef, useState } from "react";

type Props = {
  onSample: (pixels: Float32Array) => void;
  onClear?: () => void;
  disabled?: boolean;
};

const MODEL_SIZE = 96; // 96x96x3 input

/**
 * Converts an uploaded image File into a Float32Array of shape [96, 96, 3],
 * ready for inference in lib/model.ts.
 */
function fileToModelInput(
  file: File
): Promise<{ pixels: Float32Array; previewUrl: string }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      const off = document.createElement("canvas");
      off.width = MODEL_SIZE;
      off.height = MODEL_SIZE;
      const ctx = off.getContext("2d")!;

      // Neutral mid-grey background for letterbox margin
      ctx.fillStyle = "#808080";
      ctx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);

      // Letterbox: fit preserving aspect ratio, centered
      const scale = Math.min(MODEL_SIZE / img.width, MODEL_SIZE / img.height);
      const w = img.width * scale;
      const h = img.height * scale;
      ctx.drawImage(img, (MODEL_SIZE - w) / 2, (MODEL_SIZE - h) / 2, w, h);

      const { data } = ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
      const numPixels = MODEL_SIZE * MODEL_SIZE;
      const pixels = new Float32Array(numPixels * 3);

      for (let i = 0; i < numPixels; i++) {
        const r = data[i * 4];
        const g = data[i * 4 + 1];
        const b = data[i * 4 + 2];
        // MobileNet normalization: [-1, 1]
        pixels[i * 3]     = r / 127.5 - 1.0;
        pixels[i * 3 + 1] = g / 127.5 - 1.0;
        pixels[i * 3 + 2] = b / 127.5 - 1.0;
      }

      resolve({ pixels, previewUrl: url });
    };
    img.onerror = reject;
    img.src = url;
  });
}

export default function UploadSwatch({ onSample, onClear, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback(
    async (file: File | undefined) => {
      if (!file || !file.type.startsWith("image/") || disabled) return;
      try {
        const { pixels, previewUrl } = await fileToModelInput(file);
        setPreview(previewUrl);
        onSample(pixels);
      } catch (e) {
        console.error("Error processing image:", e);
      }
    },
    [onSample, disabled]
  );

  const handleClear = useCallback(() => {
    setPreview(null);
    if (inputRef.current) inputRef.current.value = "";
    if (onClear) onClear();
  }, [onClear]);

  const loadSampleUrl = useCallback(
    async (url: string) => {
      try {
        const res = await fetch(url);
        const blob = await res.blob();
        const file = new File([blob], "sample.jpg", { type: "image/jpeg" });
        await handleFile(file);
      } catch (err) {
        console.error("Failed to load sample:", err);
      }
    },
    [handleFile]
  );

  return (
    <div className="flex w-full flex-col gap-3">
      <div
        className={`stitched relative flex aspect-square w-full max-w-sm mx-auto flex-col items-center justify-center overflow-hidden bg-paper shadow-swatch transition-all duration-200 ${
          dragOver ? "border-denim bg-canvas/60 scale-[1.01]" : ""
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFile(e.dataTransfer.files?.[0]);
        }}
      >
        {preview ? (
          <div className="relative flex h-full w-full items-center justify-center p-4">
            <img
              src={preview}
              alt="Uploaded garment"
              className="max-h-full max-w-full rounded-sm object-contain shadow-sm"
            />
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center gap-3 p-6 text-center">
            <svg
              className="h-9 w-9 text-muted/70 transition-transform group-hover:scale-105"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              strokeWidth="1.5"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
              />
            </svg>
            <div className="space-y-1">
              <p className="font-tag text-xs uppercase tracking-tagwide text-muted">
                Drop garment photo, or
              </p>
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                disabled={disabled}
                className="focus-ring font-tag text-xs uppercase tracking-tagwide text-denim underline decoration-dotted underline-offset-4 transition-colors hover:text-ink"
              >
                Browse file
              </button>
            </div>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
      </div>

      {/* Quick Test Samples */}
      <div className="flex flex-wrap items-center justify-center gap-2 pt-1">
        <span className="font-tag text-[11px] uppercase tracking-wider text-muted">Try sample:</span>
        <button
          type="button"
          id="sample-tshirt-btn"
          onClick={() => loadSampleUrl("/samples/tshirt.jpg")}
          disabled={disabled}
          className="rounded border border-denim/40 bg-paper px-2.5 py-1 font-tag text-[11px] uppercase tracking-wider text-denim shadow-xs hover:bg-canvas transition-colors"
        >
          T-Shirt
        </button>
        <button
          type="button"
          id="sample-sneaker-btn"
          onClick={() => loadSampleUrl("/samples/sneaker.jpg")}
          disabled={disabled}
          className="rounded border border-denim/40 bg-paper px-2.5 py-1 font-tag text-[11px] uppercase tracking-wider text-denim shadow-xs hover:bg-canvas transition-colors"
        >
          Sneaker
        </button>
      </div>

      {preview && (
        <div className="flex items-center justify-center gap-4">
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="focus-ring font-tag text-xs uppercase tracking-tagwide text-denim underline decoration-dotted underline-offset-4 hover:text-ink"
          >
            Change photo
          </button>
          <span className="text-muted/40">•</span>
          <button
            type="button"
            onClick={handleClear}
            className="focus-ring font-tag text-xs uppercase tracking-tagwide text-muted underline decoration-dotted underline-offset-4 hover:text-thread"
          >
            Clear swatch
          </button>
        </div>
      )}
    </div>
  );
}
