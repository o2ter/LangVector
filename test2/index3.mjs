
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

console.log({ ...model.tokens });

const tokens = [
  'EOS',
  'EOT',
  '<|start_header_id|>',
  '<|end_header_id|>',
  '<|eot_id|>',
  '<|end_of_text|>',
];

for (const token of tokens) {
  console.log(token, model.tokenize(token, { encodeSpecial: true }));
}
