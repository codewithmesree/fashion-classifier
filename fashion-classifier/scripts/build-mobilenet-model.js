/**
 * build-mobilenet-model.js
 * ------------------------
 * Constructs a 96×96×3 MobileNet model with pretrained ImageNet weights transferred
 * directly into the convolutional backbone, and mapped classification weights for
 * our 10 target garment classes.
 *
 * Saves directly to public/model/model.json + shards.
 */

const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const path = require('path');

const OUT_DIR = path.resolve(__dirname, '..', 'public', 'model');

// The 10 target garment classes
const CLASS_NAMES = [
  "T-shirt/top",  // 0
  "Trouser",      // 1
  "Pullover",     // 2
  "Dress",        // 3
  "Coat",         // 4
  "Sandal",       // 5
  "Shirt",        // 6
  "Sneaker",      // 7
  "Bag",          // 8
  "Ankle boot",   // 9
];

// Mappings from ImageNet 1000 class indices to our 10 classes
const IMAGENET_CLASS_MAPPING = {
  0: [610],                  // T-shirt/top: jersey
  1: [608],                  // Trouser: jean / denim
  2: [841, 474, 264],        // Pullover: sweatshirt, cardigan
  3: [578, 400, 601, 614, 655, 689], // Dress: gown, academic gown, hoopskirt, kimono, miniskirt, overskirt
  4: [869, 568, 617, 834],   // Coat: trench coat, fur coat, lab coat, suit
  5: [774, 502],             // Sandal: sandal, clog
  6: [610],                  // Shirt: jersey / formal shirt
  7: [770, 630, 806],        // Sneaker: running shoe, loafer, sock
  8: [748, 414, 636, 728],   // Bag: purse, backpack, mailbag, plastic bag
  9: [514],                  // Ankle boot: cowboy boot
};

async function main() {
  console.log('Loading pretrained MobileNet v1 (0.25 depth multiplier)...');
  const srcModel = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  console.log('Pretrained model loaded successfully.');

  // Create a new functional / sequential model with input [96, 96, 3]
  const input = tf.input({ shape: [96, 96, 3], name: 'input_image' });
  
  let x = input;
  
  // Layer list up to conv_pw_13_relu
  // We recreate each layer with the exact same architecture, but accepting [96, 96, 3]
  const layerDefs = [
    { name: 'conv1', type: 'conv2d', filters: 8, kernelSize: 3, strides: 2, padding: 'same', useBias: false },
    { name: 'conv1_bn', type: 'bn' },
    { name: 'conv1_relu', type: 'relu' },

    { name: 'conv_dw_1', type: 'dw', kernelSize: 3, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_dw_1_bn', type: 'bn' },
    { name: 'conv_dw_1_relu', type: 'relu' },
    { name: 'conv_pw_1', type: 'conv2d', filters: 16, kernelSize: 1, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_pw_1_bn', type: 'bn' },
    { name: 'conv_pw_1_relu', type: 'relu' },

    { name: 'conv_dw_2', type: 'dw', kernelSize: 3, strides: 2, padding: 'same', useBias: false },
    { name: 'conv_dw_2_bn', type: 'bn' },
    { name: 'conv_dw_2_relu', type: 'relu' },
    { name: 'conv_pw_2', type: 'conv2d', filters: 32, kernelSize: 1, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_pw_2_bn', type: 'bn' },
    { name: 'conv_pw_2_relu', type: 'relu' },

    { name: 'conv_dw_3', type: 'dw', kernelSize: 3, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_dw_3_bn', type: 'bn' },
    { name: 'conv_dw_3_relu', type: 'relu' },
    { name: 'conv_pw_3', type: 'conv2d', filters: 32, kernelSize: 1, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_pw_3_bn', type: 'bn' },
    { name: 'conv_pw_3_relu', type: 'relu' },

    { name: 'conv_dw_4', type: 'dw', kernelSize: 3, strides: 2, padding: 'same', useBias: false },
    { name: 'conv_dw_4_bn', type: 'bn' },
    { name: 'conv_dw_4_relu', type: 'relu' },
    { name: 'conv_pw_4', type: 'conv2d', filters: 64, kernelSize: 1, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_pw_4_bn', type: 'bn' },
    { name: 'conv_pw_4_relu', type: 'relu' },

    { name: 'conv_dw_5', type: 'dw', kernelSize: 3, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_dw_5_bn', type: 'bn' },
    { name: 'conv_dw_5_relu', type: 'relu' },
    { name: 'conv_pw_5', type: 'conv2d', filters: 64, kernelSize: 1, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_pw_5_bn', type: 'bn' },
    { name: 'conv_pw_5_relu', type: 'relu' },

    { name: 'conv_dw_6', type: 'dw', kernelSize: 3, strides: 2, padding: 'same', useBias: false },
    { name: 'conv_dw_6_bn', type: 'bn' },
    { name: 'conv_dw_6_relu', type: 'relu' },
    { name: 'conv_pw_6', type: 'conv2d', filters: 128, kernelSize: 1, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_pw_6_bn', type: 'bn' },
    { name: 'conv_pw_6_relu', type: 'relu' },

    { name: 'conv_dw_7', type: 'dw', kernelSize: 3, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_dw_7_bn', type: 'bn' },
    { name: 'conv_dw_7_relu', type: 'relu' },
    { name: 'conv_pw_7', type: 'conv2d', filters: 128, kernelSize: 1, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_pw_7_bn', type: 'bn' },
    { name: 'conv_pw_7_relu', type: 'relu' },

    { name: 'conv_dw_8', type: 'dw', kernelSize: 3, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_dw_8_bn', type: 'bn' },
    { name: 'conv_dw_8_relu', type: 'relu' },
    { name: 'conv_pw_8', type: 'conv2d', filters: 128, kernelSize: 1, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_pw_8_bn', type: 'bn' },
    { name: 'conv_pw_8_relu', type: 'relu' },

    { name: 'conv_dw_9', type: 'dw', kernelSize: 3, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_dw_9_bn', type: 'bn' },
    { name: 'conv_dw_9_relu', type: 'relu' },
    { name: 'conv_pw_9', type: 'conv2d', filters: 128, kernelSize: 1, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_pw_9_bn', type: 'bn' },
    { name: 'conv_pw_9_relu', type: 'relu' },

    { name: 'conv_dw_10', type: 'dw', kernelSize: 3, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_dw_10_bn', type: 'bn' },
    { name: 'conv_dw_10_relu', type: 'relu' },
    { name: 'conv_pw_10', type: 'conv2d', filters: 128, kernelSize: 1, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_pw_10_bn', type: 'bn' },
    { name: 'conv_pw_10_relu', type: 'relu' },

    { name: 'conv_dw_11', type: 'dw', kernelSize: 3, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_dw_11_bn', type: 'bn' },
    { name: 'conv_dw_11_relu', type: 'relu' },
    { name: 'conv_pw_11', type: 'conv2d', filters: 128, kernelSize: 1, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_pw_11_bn', type: 'bn' },
    { name: 'conv_pw_11_relu', type: 'relu' },

    { name: 'conv_dw_12', type: 'dw', kernelSize: 3, strides: 2, padding: 'same', useBias: false },
    { name: 'conv_dw_12_bn', type: 'bn' },
    { name: 'conv_dw_12_relu', type: 'relu' },
    { name: 'conv_pw_12', type: 'conv2d', filters: 256, kernelSize: 1, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_pw_12_bn', type: 'bn' },
    { name: 'conv_pw_12_relu', type: 'relu' },

    { name: 'conv_dw_13', type: 'dw', kernelSize: 3, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_dw_13_bn', type: 'bn' },
    { name: 'conv_dw_13_relu', type: 'relu' },
    { name: 'conv_pw_13', type: 'conv2d', filters: 256, kernelSize: 1, strides: 1, padding: 'same', useBias: false },
    { name: 'conv_pw_13_bn', type: 'bn' },
    { name: 'conv_pw_13_relu', type: 'relu' },
  ];

  const instantiatedLayers = [];

  for (const def of layerDefs) {
    let l;
    if (def.type === 'conv2d') {
      l = tf.layers.conv2d({ name: def.name, filters: def.filters, kernelSize: def.kernelSize, strides: def.strides, padding: def.padding, useBias: def.useBias });
    } else if (def.type === 'dw') {
      l = tf.layers.depthwiseConv2d({ name: def.name, kernelSize: def.kernelSize, strides: def.strides, padding: def.padding, useBias: def.useBias });
    } else if (def.type === 'bn') {
      l = tf.layers.batchNormalization({ name: def.name });
    } else if (def.type === 'relu') {
      l = tf.layers.reLU({ name: def.name, maxValue: 6 });
    }
    x = l.apply(x);
    instantiatedLayers.push({ def, layer: l });
  }

  // Global Average Pooling
  const gap = tf.layers.globalAveragePooling2d({ name: 'global_avg_pool', dataFormat: 'channelsLast' });
  x = gap.apply(x);

  // Classification Head: Dense layer with 10 classes + softmax
  const denseHead = tf.layers.dense({ name: 'garment_classifier', units: 10, activation: 'softmax' });
  const output = denseHead.apply(x);

  const model = tf.model({ inputs: input, outputs: output, name: 'fashion_mobilenet_96' });
  console.log('Constructed 96x96 MobileNet model architecture.');

  // Transfer weights from pretrained MobileNet into our instantiated layers
  console.log('Transferring pretrained backbone weights...');
  for (const { def, layer } of instantiatedLayers) {
    try {
      const srcLayer = srcModel.getLayer(def.name);
      const srcWeights = srcLayer.getWeights();
      if (srcWeights.length > 0) {
        layer.setWeights(srcWeights);
      }
    } catch (e) {
      console.warn(`Could not transfer weights for ${def.name}:`, e.message);
    }
  }
  console.log('Backbone weights transferred.');

  // Build the 10-class classifier weights using the 1000-class conv_preds
  const convPreds = srcModel.getLayer('conv_preds');
  const [srcKernel, srcBias] = convPreds.getWeights();
  const kData = srcKernel.dataSync(); // shape [1, 1, 256, 1000]
  const bData = srcBias.dataSync();   // shape [1000]

  const newKernel = new Float32Array(256 * 10);
  const newBias = new Float32Array(10);

  for (let c = 0; c < 10; c++) {
    const indices = IMAGENET_CLASS_MAPPING[c];
    for (let f = 0; f < 256; f++) {
      let sumK = 0;
      for (const idx of indices) {
        // kData index for [0, 0, f, idx] is f * 1000 + idx
        sumK += kData[f * 1000 + idx];
      }
      newKernel[f * 10 + c] = (sumK / indices.length) * 2.5; // slight temperature scaling for confident classification
    }
    let sumB = 0;
    for (const idx of indices) {
      sumB += bData[idx];
    }
    newBias[c] = sumB / indices.length;
  }

  const kernelTensor = tf.tensor2d(newKernel, [256, 10]);
  const biasTensor = tf.tensor1d(newBias);
  denseHead.setWeights([kernelTensor, biasTensor]);
  console.log('Garment classification head weights initialized.');

  // Save the model to public/model/
  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

  // Clean stale files
  for (const f of fs.readdirSync(OUT_DIR)) {
    if (f !== '.gitkeep') {
      fs.rmSync(path.join(OUT_DIR, f));
    }
  }

  await model.save(tf.io.withSaveHandler(async (artifacts) => {
    const modelJson = {
      modelTopology: artifacts.modelTopology,
      format: artifacts.format,
      generatedBy: artifacts.generatedBy,
      convertedBy: artifacts.convertedBy,
      weightsManifest: [{
        paths: ['./group1-shard1of1.bin'],
        weights: artifacts.weightSpecs
      }]
    };
    fs.writeFileSync(path.join(OUT_DIR, 'model.json'), JSON.stringify(modelJson, null, 2));
    fs.writeFileSync(path.join(OUT_DIR, 'group1-shard1of1.bin'), Buffer.from(artifacts.weightData));
    return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
  }));

  console.log(`\n✅  Successfully created and saved pretrained 96×96 MobileNet garment classifier to: ${OUT_DIR}`);
  console.log(`    Weights size: ${(fs.statSync(path.join(OUT_DIR, 'group1-shard1of1.bin')).size / 1e6).toFixed(2)} MB`);
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
