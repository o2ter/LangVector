import * as PDF from 'mupdf';
import { fileURLToPath } from "url";
import path from "path";
import { getLlama, LlamaChatSession, defineChatSessionFunction } from 'node-llama-cpp';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const llama = await getLlama();
const model = await llama.loadModel({
  modelPath: path.join(__dirname, 'models/meta-llama/Meta-Llama-3-8B/ggml-model-q3_k_m.gguf')
});

console.log(`systemInfo: ${llama.systemInfo}`);
console.log(`GPU: ${llama.gpu}`);
console.log(`GPU Devices: ${await llama.getGpuDeviceNames()}`);

const context = await model.createContext();
const session = new LlamaChatSession({
  contextSequence: context.getSequence()
});

const options = {
  functions: {
    random: defineChatSessionFunction({
      description: "Generates a random number",
      handler() {
        console.log('check random called')
        return Math.random();
      }
    })
  }
};

const questions = [
  'Hi there, how are you?',
  'Give me a random number',
];

for (const question of questions) {
  const cache = [];

  console.log("User: " + question);

  const ans = await session.promptWithMeta(question, {
    ...options,
    temperature: 0.8,
    onToken: (token) => {
      cache.push(...token)
      console.log(model.detokenize(cache, true))
    }
  });

  console.log("AI: ", ans);
}
