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
import { defineChatSessionFunction, llamaCpp, LLMDevice } from './dist/index.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const defaultOptions = {
  documentFunctionParams: true,
  functions: {
    datetime: defineChatSessionFunction({
      description: "Get current datetime",
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
      handler({ maximum, minimum }) {
        return Math.random() * (maximum - minimum) + minimum;
      }
    }),
    todayMenu: defineChatSessionFunction({
      description: "A list of today’s special menu",
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
  }
};

class ChatWrapper extends llamaCpp.Llama3ChatWrapper {

  generateAvailableFunctionsSystemText(
    availableFunctions,
    { documentParams = true },
  ) {
    const result = super.generateAvailableFunctionsSystemText(availableFunctions, { documentParams });
    return result.mapValues(s => _.isString(s) ? s.replace('Note that the || prefix is mandatory', 'Note that the ||call: prefix is mandatory') : s);
  }
}

const device = await LLMDevice.llama();

const model = await device.loadModel({
  modelPath: path.join(__dirname, 'models', 'meta-llama/Meta-Llama-3-8B-Instruct/ggml-model-q3_k_m.gguf'),
  ignoreMemorySafetyChecks: true,
});

const context = await model.createContext({ ignoreMemorySafetyChecks: true });
const session = await context.createSession({
  chatOptions: { chatWrapper: new ChatWrapper }
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
];

for (const quest of quests) {

  let partial = [];
  const { responseText } = await session.prompt(quest, {
    ...defaultOptions,
    ...options,
    onToken: (token) => {
      partial.push(...token);
      console.log(model.detokenize(partial, true));
    }
  });

  console.log(responseText);
  console.log('');
}

console.log(model.detokenize(session.tokens, true));
