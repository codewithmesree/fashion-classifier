import * as tf from "@tensorflow/tfjs";

let modelPromise: Promise<tf.LayersModel> | null = null;

/**
 * Loads /public/model/model.json once and caches the promise.
 * Run scripts/train_mobilenet.py to train + export the MobileNetV2 model,
 * which writes model.json + weight shard files into public/model/ automatically.
 */
export function loadModel(): Promise<tf.LayersModel> {
  if (!modelPromise) {
    modelPromise = tf.loadLayersModel("/model/model.json");
  }
  return modelPromise;
}

/**
 * Runs inference on a preprocessed 96×96 RGB image.
 * `pixels` must be length 27,648 (96 × 96 × 3), each value normalized to
 * [-1, 1] using MobileNetV2's convention: (pixel_0_255 / 127.5) − 1.0.
 * Channels are interleaved in row-major order: [R,G,B, R,G,B, …].
 */
export async function predict(pixels: Float32Array): Promise<number[]> {
  const model = await loadModel();
  return tf.tidy(() => {
    const input = tf.tensor4d(pixels, [1, 96, 96, 3]);
    const output = model.predict(input) as tf.Tensor;
    return Array.from(output.dataSync());
  });
}
