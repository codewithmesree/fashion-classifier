/**
 * generate-stub-model.js
 * ----------------------
 * Builds and saves a lightweight MobileNetV2-compatible model with the correct
 * 96×96×3 input shape and 10-class output.
 *
 * This is NOT a trained model — weights are randomly initialised. It is
 * intended only to:
 *   1. Confirm the frontend pipeline works end-to-end (no shape errors).
 *   2. Serve as a drop-in placeholder until train_mobilenet.py has run.
 *
 * Usage:
 *   node scripts/generate-stub-model.js
 *
 * Output:
 *   public/model/model.json
 *   public/model/group1-shard1of1.bin   (or multiple shards)
 */

const tf   = require('@tensorflow/tfjs-node');
const path = require('path');
const fs   = require('fs');

const IMG_SIZE    = 96;
const NUM_CLASSES = 10;
const OUT_DIR     = path.resolve(__dirname, '..', 'public', 'model');

function buildStubModel() {
  // A small CNN that exactly mirrors the input/output contract of the real
  // MobileNetV2 model: input [1, 96, 96, 3] normalised to [-1, 1],
  // output [1, 10] softmax.
  const model = tf.sequential({ name: 'fashion_mobilenetv2_stub' });

  // Block 1
  model.add(tf.layers.conv2d({
    inputShape: [IMG_SIZE, IMG_SIZE, 3],
    filters: 32, kernelSize: 3, strides: 2, padding: 'same', activation: 'relu',
  }));

  // Block 2
  model.add(tf.layers.depthwiseConv2d({
    kernelSize: 3, padding: 'same', activation: 'relu',
  }));
  model.add(tf.layers.conv2d({ filters: 64, kernelSize: 1, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  // Block 3
  model.add(tf.layers.depthwiseConv2d({
    kernelSize: 3, padding: 'same', activation: 'relu',
  }));
  model.add(tf.layers.conv2d({ filters: 128, kernelSize: 1, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  // Block 4
  model.add(tf.layers.depthwiseConv2d({
    kernelSize: 3, padding: 'same', activation: 'relu',
  }));
  model.add(tf.layers.conv2d({ filters: 256, kernelSize: 1, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  // Head
  model.add(tf.layers.globalAveragePooling2d({ dataFormat: 'channelsLast' }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' }));

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

async function main() {
  console.log('Building stub model (96×96×3 → 10 classes)…');
  const model = buildStubModel();
  model.summary();

  console.log('Model built successfully.');

  // Save to public/model/
  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

  // Remove stale shards from the old 28×28 model
  for (const f of fs.readdirSync(OUT_DIR)) {
    if (f !== '.gitkeep') {
      fs.rmSync(path.join(OUT_DIR, f));
      console.log(`Removed stale file: ${f}`);
    }
  }

  const savePath = `file://${OUT_DIR}`;
  await model.save(savePath);

  const files = fs.readdirSync(OUT_DIR).filter(f => f !== '.gitkeep');
  const totalBytes = files.reduce(
    (sum, f) => sum + fs.statSync(path.join(OUT_DIR, f)).size, 0
  );
  console.log(`\n✅  Saved to ${OUT_DIR}`);
  console.log(`   Files: ${files.join(', ')}`);
  console.log(`   Total: ${(totalBytes / 1e6).toFixed(2)} MB`);
  console.log('\n⚠  This is a STUB — weights are random, predictions are meaningless.');
  console.log('   Replace with the real model after running:');
  console.log('     python scripts/train_mobilenet.py');
}

main().catch(err => { console.error(err); process.exit(1); });
