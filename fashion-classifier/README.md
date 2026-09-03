# Garment Scanner — Fashion-MNIST Classifier

A Next.js frontend for your Fashion-MNIST CNN. Inference runs entirely
client-side with TensorFlow.js — no backend, no server cost, deploys to
Vercel as a static app.

Design concept: the UI is styled like a garment care tag — a stitched-edge
fabric swatch for input, a hang tag (with grommet) for the top prediction,
and ribbon-style bars for the full probability breakdown.

## 1. Export the trained model

In the notebook, after training finishes (after the `model.fit` cell), add:

```python
model.save("fashion_model.keras")
```

Then, in a terminal (a fresh virtualenv is safest — `tensorflowjs` pins
specific TF versions):

```bash
pip install tensorflowjs
tensorflowjs_converter --input_format=keras \
  fashion_model.keras \
  ./model_output
```

Copy everything `tensorflowjs_converter` produced (`model.json` and the
`group1-shard*.bin` weight files) into `public/model/` in this project,
replacing the `.gitkeep` placeholder.

**Sanity-check before trusting the UI:** the drawing canvas inverts
strokes (dark-on-light → light-on-dark) to match how Fashion-MNIST images
are encoded. Pull a real image out of `x_test` in the notebook, run it
through the same preprocessing your browser code uses, and confirm the
prediction matches `model.predict()` in Python. If confidences look
scrambled, that inversion (in `components/DrawCanvas.tsx`) or the
upload-path normalization (in `components/UploadSwatch.tsx`) is the first
place to check.

## 2. Run locally

```bash
npm install
npm run dev
```

Open http://localhost:3000.

## 3. Deploy to Vercel

```bash
npm install -g vercel   # or use the Vercel dashboard
vercel
```

Or via the dashboard: push this folder to a GitHub repo, then
**Import Project** on vercel.com and select it. Next.js is auto-detected,
no build settings needed. The model files in `public/model/` are served
as static assets automatically.

## Project structure

```
app/page.tsx              — composes the input + results layout
app/layout.tsx            — fonts (Space Grotesk / Inter / IBM Plex Mono)
components/DrawCanvas.tsx — touch-enabled drawing input, 28x28 preprocessing
components/UploadSwatch.tsx — drag-and-drop image upload, alternate input
components/GarmentTag.tsx — signature "hang tag" prediction display
components/PredictionPanel.tsx — ribbon-bar breakdown across all 10 classes
lib/model.ts               — tf.js model loading + inference
lib/classes.ts              — class name list, must match training order
```
