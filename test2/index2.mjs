
import _ from 'lodash';
import path from 'path';
import { fileURLToPath } from 'url';
import { getLlama } from 'node-llama-cpp';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const llama = await getLlama();
const model2 = await llama.loadModel({
  modelPath: path.join(__dirname, '../models', 'meta-llama/Meta-Llama-3-8B-Instruct/ggml-model-q3_k_m.gguf'),
});
const context2 = await model2.createEmbeddingContext({});

console.log(await context2.getEmbeddingFor('hello, world'))
