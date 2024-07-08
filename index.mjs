import _ from "lodash";
import { fileURLToPath } from "url";
import path from "path";
import { getLlama } from 'node-llama-cpp';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const llama = await getLlama();
const model = await llama.loadModel({
  modelPath: path.join(__dirname, 'models/meta-llama/Meta-Llama-3-8B-Instruct/ggml-model-q3_k_m.gguf')
});

console.log(`systemInfo: ${llama.systemInfo}`);
console.log(`GPU: ${llama.gpu}`);
console.log(`GPU Devices: ${await llama.getGpuDeviceNames()}`);

const embeddingContext = await model.createEmbeddingContext();

const text = "Hello world";
const embedding = await embeddingContext.getEmbeddingFor(text);

console.log(embedding.vector);
console.log(embedding.vector.length);

