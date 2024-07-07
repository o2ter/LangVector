import * as PDF from 'mupdf';
import { fileURLToPath } from "url";
import path from "path";
import { getLlama, LlamaChatSession, defineChatSessionFunction } from 'node-llama-cpp';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const llama = await getLlama();
const model = await llama.loadModel({
  modelPath: path.join(__dirname, 'models/meta-llama/Meta-Llama-3-8B/ggml-model-q5_k_m.gguf')
});

const context = await model.createContext();
const session = new LlamaChatSession({
  contextSequence: context.getSequence()
});

const options = {
  functions: {
    random: defineChatSessionFunction({
      description: "Generates a random number",
      handler() {
        return Math.random();
      }
    })
  }
};

const q1 = "Hi there, how are you?";
console.log("User: " + q1);

let cache = []

const a1 = await session.prompt(q1, {
  ...options,
  onToken: (token) => {
    cache.push(...token)
    console.log(model.detokenize(cache, true))
  }
});
console.log("AI: " + a1);
