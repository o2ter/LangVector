//
//  session.ts
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
import { defineChatSessionFunction, Llama3ChatWrapper, LlamaDevice, LlamaModel } from '../../src';

export const modelsDir = path.join(__dirname, '../../models');

export const functions = {
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
};

const models: Record<string, LlamaModel> = {};

export const createModel = async (name: string) => {
  if (models[name]) return models[name];
  const model = await LlamaDevice.loadModel({
    modelPath: path.join(modelsDir, name),
  });
  models[name] = model;
  return model;
}

export const createContext = async (name: string) => {
  const model = await createModel(name);
  return model.createContext({
    contextSize: 6752,
    chatOptions: {
      chatWrapper: new Llama3ChatWrapper,
      functions,
    },
  });
}
