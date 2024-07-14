
import _ from 'lodash';
import path from 'path';
import { fileURLToPath } from 'url';
import { LlamaDevice } from '../dist/index.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const model = await LlamaDevice.loadModel({
  modelPath: path.join(__dirname, '../models', 'meta-llama/Meta-Llama-3-8B-Instruct/ggml-model-q3_k_m.gguf'),
  useMmap: true,
});

let t = Date.now()

const context = model.createContext({
  embeddings: true,
  contextSize: 4096,
});

const result = await context.embedding('hello, world')

const t2 = Date.now() - t;

console.log(result, t2 / 1000)
