
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

for await (const { response, ...rest } of context.prompt('你好，你叫咩名？', { temperature: 0.8, maxTokens: 500 })) {
  console.log({ ...rest, text: model.detokenize(response, { decodeSpecial: true }) });
}

for await (const { response, ...rest } of context.prompt('hi', { temperature: 0.8, maxTokens: 500 })) {
  console.log({ ...rest, text: model.detokenize(response, { decodeSpecial: true }) });
}

for await (const { response, ...rest } of context.prompt('請你提供一個json 例子', { temperature: 0.8, maxTokens: 500 })) {
  console.log({ ...rest, text: model.detokenize(response, { decodeSpecial: true }) });
}

console.log('------------------------------------------');
console.log(model.detokenize(context.tokens, { decodeSpecial: true }));
console.log('------------------------------------------');
console.log(context.chatHistory);
