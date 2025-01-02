//
//  session.ts
//
//  The MIT License
//  Copyright (c) 2021 - 2025 O2ter Limited. All rights reserved.
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
import fs from 'fs/promises';
import { defineChatSessionFunction, Llama3ChatWrapper, LlamaDevice, LlamaModel } from '../../src';

const walkDirAsync = async function* (dir: string): AsyncGenerator<string, void> {
  const files = await fs.readdir(dir, { withFileTypes: true });
  for (const file of files) {
    if (file.isDirectory()) {
      yield* walkDirAsync(path.join(dir, file.name));
    } else {
      yield path.join(dir, file.name);
    }
  }
}

export const modelsDir = path.join(__dirname, '../../models');

export const models: string[] = [];

try {
  for await (const file of walkDirAsync(modelsDir)) {
    if (file.endsWith('.gguf')) {
      models.push(file.slice(modelsDir.length + 1));
    }
  }
} catch { }

export const functions = {
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
      },
      required: ['totalCount', 'menus'],
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

const modelCaches: Record<string, LlamaModel> = {};

export const createModel = async (name: string) => {
  if (modelCaches[name]) return modelCaches[name];
  const model = await LlamaDevice.loadModel({
    modelPath: path.join(modelsDir, name),
  });
  modelCaches[name] = model;
  return model;
}

export const createContext = async (name: string) => {
  const model = await createModel(name);
  return model.createContext({
    contextSize: 4096,
    chatOptions: {
      chatWrapper: new Llama3ChatWrapper,
      functions,
    },
  });
}
