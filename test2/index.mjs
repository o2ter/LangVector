
import _ from 'lodash';
import path from 'path';
import { fileURLToPath } from 'url';
import { LlamaDevice, Llama3ChatWrapper } from '../dist/index.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const model = await LlamaDevice.loadModel({
  modelPath: path.join(__dirname, '../models', 'meta-llama/Meta-Llama-3-8B-Instruct/ggml-model-q3_k_m.gguf'),
  useMmap: true,
});

// console.log(await model.embedding('hello, world'));

const context = model.createContext({
  contextSize: 512,
  chatOptions: {
    chatWrapper: new Llama3ChatWrapper,
  },
});

const tokenBias = new Map;
tokenBias.set(0, 'never');

for await (const { token, time } of context.prompt('hello, world', { temperature: 0.8, maxTokens: 500, tokenBias })) {
  console.log({ token, time, text: model.detokenize(token, { decodeSpecial: true }) })
}

console.log(model.detokenize(context.tokens, { decodeSpecial: true }))
