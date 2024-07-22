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
import { Llama3ChatWrapper, LlamaDevice, defineChatSessionFunction } from './dist/index.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const functions = {
  datetime: defineChatSessionFunction({
    description: "Get current ISO datetime in UTC",
    resultType: { type: 'string' },
    handler() {
      return new Date();
    }
  }),
  randomInt: defineChatSessionFunction({
    description: "Generates a random integer between maximum and minimum inclusively",
    params: {
      type: 'object',
      properties: {
        maximum: { type: 'integer' },
        minimum: { type: 'integer' },
      },
      required: ['maximum', 'minimum'],
    },
    resultType: { type: 'integer' },
    handler({ maximum, minimum }) {
      return Math.floor(Math.random() * (maximum - minimum + 1)) + minimum;
    }
  }),
  randomFloat: defineChatSessionFunction({
    description: "Generates a random floating number between maximum and minimum",
    params: {
      type: 'object',
      properties: {
        maximum: { type: 'number' },
        minimum: { type: 'number' },
      },
      required: ['maximum', 'minimum'],
    },
    resultType: { type: 'number' },
    handler({ maximum, minimum }) {
      return Math.random() * (maximum - minimum) + minimum;
    }
  }),
  todayMenu: defineChatSessionFunction({
    description: "A list of today’s special menu",
    resultType: {
      type: 'object',
      properties: {
        totalCount: { type: 'integer' },
        menus: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              name: { type: 'string' },
              price: { type: 'number' },
            },
            required: ['name', 'price'],
          },
        },
        required: ['totalCount', 'menus'],
      },
      required: ['maximum', 'minimum'],
    },
    handler() {
      return {
        totalCount: 3,
        menus: [
          {
            name: 'Pizza',
            price: 75,
          },
          {
            name: 'Hamburger',
            price: 80,
          },
          {
            name: 'Fish And Chips',
            price: 75,
          },
        ],
      };
    }
  }),
};

const model = await LlamaDevice.loadModel({
  modelPath: path.join(__dirname, 'models', 'meta-llama/Meta-Llama-3-8B-Instruct/ggml-model-q3_k_m.gguf'),
  useMmap: true,
});

const context = await model.createContext({
  contextSize: 4096,
  chatOptions: {
    chatWrapper: new Llama3ChatWrapper,
    functions,
  },
});

const options = {
  minP: 0,
  topK: 40,
  topP: 0.75,
  temperature: 0.8,
  repeatPenalty: {
    frequencyPenalty: 0.2,
    presencePenalty: 0.2,
  },
  maxTokens: 100,
};

const quests = [
  'Hi',
  'Can you pick one item from menu randomly?',
  'What is the time of Hong Kong now?',
  'What is time offset in Hong Kong?',
];

for (const quest of quests) {

  const generator = context.prompt(quest, {
    ...options,
  });
  for await (const { response, ...rest } of generator) {
    console.log({ ...rest, text: model.detokenize(response, { decodeSpecial: true }) });
  }

  console.log('');
}

console.log(model.detokenize(context.tokens, { decodeSpecial: true }));
