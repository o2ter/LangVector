import * as PDF from 'mupdf';
import { fileURLToPath } from "url";
import path from "path";
import { getLlama } from 'node-llama-cpp';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const llama = await getLlama();
const model = await llama.loadModel({
  modelPath: path.join(__dirname, 'models/Trelis/Meta-Llama-3-8B-Instruct-function-calling/ggml-model-q5_k_m.gguf')
});

const embeddingContext = await model.createEmbeddingContext();

const text = "Hello world";
const embedding = await embeddingContext.getEmbeddingFor(text);

console.log(embedding.vector);
console.log(embedding.vector.length);

