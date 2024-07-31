//
//  test.mjs
//
//  The MIT License
//  Copyright (c) 2021 - 2024 O2ter Limited. All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//

import _ from 'lodash';
import path from 'path';
import { fileURLToPath } from 'url';
import { Llama3ChatWrapper, LlamaDevice, gbnf } from './dist/index.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const model = await LlamaDevice.loadModel({
  modelPath: path.join(__dirname, 'models', 'meta-llama/Meta-Llama-3.1-8B-Instruct/ggml-model-q3_k_m.gguf'),
  useMmap: true,
});

const questions = [
  'What\'s your name?',
  'Where am I?',
];

class ChatWrapper extends Llama3ChatWrapper {

  generateSystemMessage(ctx) {
    return [
      'You are a helpful, respectful and honest assistant.',
      'You will be provided list of questions. Pick the best match question from the choices.',
      'You only need to tell the index of the question.',
      'If none of the question matched, just say 0.',
      'Do not add any other unnecessary content in your response.',
      '',
      'These are the provided questions:',
      ..._.map(questions, (x, i) => `${i + 1}. ${x}`),
    ].join('\n');
  }
}

const context = await model.createContext({
  contextSize: 4096,
  chatOptions: {
    chatWrapper: new ChatWrapper,
  },
});

const options = {
  minP: 0,
  topK: 40,
  topP: 0.75,
  temperature: 0.05,
  repeatPenalty: {
    frequencyPenalty: 0.2,
    presencePenalty: 0.2,
  },
  maxTokens: 100,
};

const grammar = gbnf`[0] | [1-9] [0-9]{0,15}`;

const quests = [
  'Hi',
  '你叫咩名?',
  'What is the time of Hong Kong now?',
  'What is time offset in Hong Kong?',
  '頂你呀',
  'What\'s your favorite color?',
];

for (const quest of quests) {

  const generator = context.prompt(quest, {
    ...options,
    grammar: new LlamaDevice.Grammar(grammar.toString()),
  });
  for await (const { response, ...rest } of generator) {
    console.log({ ...rest, text: model.detokenize(response, { decodeSpecial: true }) });
  }

  console.log('');
}

console.log(model.detokenize(context.tokens, { decodeSpecial: true }));
