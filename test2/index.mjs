
import _ from 'lodash';
import path from 'path';
import { fileURLToPath } from 'url';
import { LlamaDevice } from '../dist/index.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log(LlamaDevice.systemInfo());
console.log(LlamaDevice.supportsGpuOffloading());
console.log(LlamaDevice.supportsMmap());
console.log(LlamaDevice.supportsMlock());
console.log(LlamaDevice.blockSizeForGgmlType(0));
console.log(LlamaDevice.typeSizeForGgmlType(0));
console.log(LlamaDevice.consts());
console.log(LlamaDevice.gpuType());
console.log(LlamaDevice.gpuDeviceInfo());
console.log(LlamaDevice.gpuVramInfo());

const model = await LlamaDevice.loadModel({
  modelPath: path.join(__dirname, '../models', 'meta-llama/Meta-Llama-3-8B-Instruct/ggml-model-q3_k_m.gguf'),
  useMmap: true,
  onLoadProgress: progress => {
    console.log({ progress });
  },
});

console.log(model.description);
console.log(model.contextSize);
console.log(model.meta);
console.log(model.meta['tokenizer.chat_template']);
console.log({ ...model.tokens });

console.log(model.tokenize('hello, world'));
console.log(model.detokenize([15339, 11, 1917]));

const context = model.createContext();
console.log(context);

const session = context.createSession();
console.log(session);
console.log(session.contextSize);