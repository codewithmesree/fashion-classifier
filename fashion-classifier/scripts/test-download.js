const http = require('http');
const https = require('https');
const zlib = require('zlib');
const fs = require('fs');

async function download(url) {
  return new Promise((resolve, reject) => {
    const client = url.startsWith('https') ? https : http;
    client.get(url, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        return download(res.headers.location).then(resolve, reject);
      }
      if (res.statusCode !== 200) {
        return reject(new Error(`Failed with status ${res.statusCode}`));
      }
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => resolve(Buffer.concat(chunks)));
      res.on('error', reject);
    });
  });
}

(async () => {
  try {
    console.log('Testing download of Fashion-MNIST labels...');
    const buf = await download('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz');
    const unzipped = zlib.gunzipSync(buf);
    console.log('Success! Unzipped size:', unzipped.length);
  } catch (err) {
    console.error('Error:', err.message);
  }
})();
