import _ from "lodash";
import { fileURLToPath } from "url";
import path from "path";
import {
  getLlama,
  LlamaChatSession,
  defineChatSessionFunction,
  Llama3ChatWrapper,
} from 'node-llama-cpp';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const llama = await getLlama();
const model = await llama.loadModel({
  modelPath: path.join(__dirname, 'models/meta-llama/Meta-Llama-3-8B-Instruct/ggml-model-q3_k_m.gguf')
});

console.log(`systemInfo: ${llama.systemInfo}`);
console.log(`GPU: ${llama.gpu}`);
console.log(`GPU Devices: ${await llama.getGpuDeviceNames()}`);

class MyChatWrapper extends Llama3ChatWrapper {
  
  generateAvailableFunctionsSystemText(availableFunctions, { documentParams = true }) {
    const result = super.generateAvailableFunctionsSystemText(availableFunctions, { documentParams });
    return result.mapValues(s => _.isString(s) ? s.replace('Note that the || prefix is mandatory', 'Note that the ||call: prefix is mandatory') : s);
  }
}

const context = await model.createContext();
const session = new LlamaChatSession({
  contextSequence: context.getSequence(),
  chatWrapper: new MyChatWrapper(),
});

const options = {
  documentFunctionParams: true,
  functions: {
    datetime: defineChatSessionFunction({
      description: "Get current datetime",
      handler() {
        console.log('function called');
        return new Date();
      }
    }),
    random: defineChatSessionFunction({
      description: "Generates a random number",
      params: {
        type: 'object',
        properties: {
          maximum: { type: 'number' },
          minimum: { type: 'number' },
        },
        required: ['maximum', 'minimum'],
      },
      handler({ maximum, minimum }) {
        console.log('function called');
        return Math.random() * (maximum - minimum) + minimum;
      }
    })
  }
};

const questions = [
  'Hi there, how are you?',
  'Give me a random number between 1 to 9',
  'What\'s the date today',
  'What\'s the time now',
];

for (const question of questions) {
  const cache = [];

  console.log("User: " + question);

  const ans = await session.promptWithMeta(question, {
    ...options,
    topK: 40,
    topP: 0.75,
    temperature: 0.8,
    repeatPenalty: {
      frequencyPenalty: 0.2,
      presencePenalty: 0.2,
    },
    maxTokens: 100,
    onToken: (token) => {
      cache.push(...token)
      console.log(model.detokenize(cache, true))
    }
  });

  console.log("AI: ", ans);
}

console.log(model.detokenize(session.sequence.contextTokens, true))
