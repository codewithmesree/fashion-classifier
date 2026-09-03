const http = require('http');
const https = require('https');
const zlib = require('zlib');
const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs');

const BASE_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/';
const FILES = {
  trainImages: 'train-images-idx3-ubyte.gz',
  trainLabels: 'train-labels-idx1-ubyte.gz',
  testImages:  't10k-images-idx3-ubyte.gz',
  testLabels:  't10k-labels-idx1-ubyte.gz'
};
const CACHE_DIR = path.resolve(__dirname, '..', '.cache');

async function getOrDownload(fileName) {
  if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR, { recursive: true });
  const filePath = path.join(CACHE_DIR, fileName);
  if (fs.existsSync(filePath)) return fs.readFileSync(filePath);
  console.log(`Downloading ${fileName}...`);
  const buf = await download(BASE_URL + fileName);
  fs.writeFileSync(filePath, buf);
  return buf;
}

async function download(url) {
  return new Promise((resolve, reject) => {
    const client = url.startsWith('https') ? https : http;
    client.get(url, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location)
        return download(res.headers.location).then(resolve, reject);
      if (res.statusCode !== 200) return reject(new Error(`HTTP ${res.statusCode}`));
      const chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => resolve(Buffer.concat(chunks)));
      res.on('error', reject);
    });
  });
}

function parseIDXImages(buffer, maxCount = null) {
  const n = maxCount ? Math.min(buffer.readUInt32BE(4), maxCount) : buffer.readUInt32BE(4);
  const out = new Float32Array(n * 784);
  for (let i = 0; i < n * 784; i++) out[i] = buffer[16 + i] / 255.0;
  return { data: out, count: n };
}

function parseIDXLabels(buffer, maxCount = null) {
  const n = maxCount ? Math.min(buffer.readUInt32BE(4), maxCount) : buffer.readUInt32BE(4);
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = buffer[8 + i];
  return { data: out, count: n };
}

// Same compact architecture as the proven original — fast matrix-multiply in JS.
// Augmentation is handled at inference time by UploadSwatch.tsx preprocessing.
function buildModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [28, 28, 1] }));
  model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 64,  activation: 'relu' }));
  model.add(tf.layers.dense({ units: 10,  activation: 'softmax' }));
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  return model;
}

async function main() {
  console.log('--- Fashion-MNIST MLP Training (8 epochs) ---');

  const [rawTI, rawTL, rawVI, rawVL] = await Promise.all([
    getOrDownload(FILES.trainImages),
    getOrDownload(FILES.trainLabels),
    getOrDownload(FILES.testImages),
    getOrDownload(FILES.testLabels)
  ]);

  const trainImgs = parseIDXImages(zlib.gunzipSync(rawTI), 30000);
  const trainLbls = parseIDXLabels(zlib.gunzipSync(rawTL), 30000);
  const testImgs  = parseIDXImages(zlib.gunzipSync(rawVI));
  const testLbls  = parseIDXLabels(zlib.gunzipSync(rawVL));

  console.log(`Train: ${trainImgs.count}, Test: ${testImgs.count}`);

  const xs     = tf.tensor4d(trainImgs.data, [trainImgs.count, 28, 28, 1]);
  const ys     = tf.oneHot(tf.tensor1d(trainLbls.data, 'int32'), 10);
  const testXs = tf.tensor4d(testImgs.data,  [testImgs.count,  28, 28, 1]);
  const testYs = tf.oneHot(tf.tensor1d(testLbls.data,  'int32'), 10);

  const model = buildModel();
  model.summary();

  console.log('\nTraining for 8 epochs (~10 min on CPU)...');
  const start = Date.now();

  await model.fit(xs, ys, {
    batchSize: 128,
    epochs: 8,
    shuffle: true,
    validationData: [testXs, testYs],
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const t = ((Date.now() - start) / 1000).toFixed(0);
        console.log(
          `Epoch ${epoch + 1}/8 [${t}s]: ` +
          `loss=${logs.loss.toFixed(4)}, acc=${(logs.acc * 100).toFixed(2)}%, ` +
          `val_loss=${logs.val_loss.toFixed(4)}, val_acc=${(logs.val_acc * 100).toFixed(2)}%`
        );
      }
    }
  });

  console.log(`\nTraining done in ${((Date.now() - start) / 1000).toFixed(1)}s`);
  xs.dispose(); ys.dispose(); testXs.dispose(); testYs.dispose();

  const targetDir = path.resolve(__dirname, '..', 'public', 'model');
  if (!fs.existsSync(targetDir)) fs.mkdirSync(targetDir, { recursive: true });

  await model.save(tf.io.withSaveHandler(async (artifacts) => {
    const modelJson = {
      modelTopology: artifacts.modelTopology,
      format: artifacts.format,
      generatedBy: artifacts.generatedBy,
      convertedBy: artifacts.convertedBy,
      weightsManifest: [{ paths: ['./group1-shard1of1.bin'], weights: artifacts.weightSpecs }]
    };
    fs.writeFileSync(path.join(targetDir, 'model.json'), JSON.stringify(modelJson, null, 2));
    fs.writeFileSync(path.join(targetDir, 'group1-shard1of1.bin'), Buffer.from(artifacts.weightData));
    console.log('Saved to public/model/');
    return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
  }));

  console.log('Done!');
}

main().catch(err => { console.error(err); process.exit(1); });
