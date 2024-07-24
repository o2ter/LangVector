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
import { LlamaDevice } from './dist/index.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const model = await LlamaDevice.loadModel({
  modelPath: path.join(__dirname, 'models', 'sentence-transformers/all-MiniLM-L6-v2/ggml-model-f16.gguf'),
  useMmap: true,
});

const test = 'That is a happy person';

const list = [
  'That is a happy dog',
  'That is a very happy person',
  'Today is a sunny day',
];

const { vector: v1 } = await model.embedding(test);

for (const str of list) {
  const { vector: v2 } = await model.embedding(str);
  const distance = Math.sqrt(_.sumBy(_.zip(v1, v2), ([a, b]) => (a - b) ** 2));
  const cosine = _.sumBy(_.zip(v1, v2), ([a, b]) => a * b) / Math.sqrt(_.sumBy(v1, v => v ** 2) * _.sumBy(v2, v => v ** 2));
  console.log({ distance, cosine })
}
