
import _ from 'lodash';
import path from 'path';
import { fileURLToPath } from 'url';
import { LlamaDevice } from '../dist/index.mjs';

import { getLlama } from 'node-llama-cpp';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const model = await LlamaDevice.loadModel({
  modelPath: path.join(__dirname, '../models', 'meta-llama/Meta-Llama-3-8B-Instruct/ggml-model-q3_k_m.gguf'),
  useMmap: true,
});

console.log('====================================')

const context = model.createContext({
  embeddings: true,
  contextSize: 512,
});

console.log(await context.embedding('hello, world'))

const llama = await getLlama();
const model2 = await llama.loadModel({
  modelPath: path.join(__dirname, '../models', 'meta-llama/Meta-Llama-3-8B-Instruct/ggml-model-q3_k_m.gguf'),
});
const context2 = await model2.createEmbeddingContext({
  contextSize: 512,
});

console.log(await context2.getEmbeddingFor('hello, world'))
