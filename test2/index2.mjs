
import _ from 'lodash';
import path from 'path';
import { fileURLToPath } from 'url';
import { getLlama, LlamaChatSession } from 'node-llama-cpp';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const llama = await getLlama();
const model = await llama.loadModel({
  modelPath: path.join(__dirname, '../models', 'meta-llama/Meta-Llama-3-8B-Instruct/ggml-model-q3_k_m.gguf'),
});
const context = await model.createContext({});
const session = new LlamaChatSession({ context });

const context2 = await model.createEmbeddingContext({});

console.log(await context2.getEmbeddingFor('hello, world'))
