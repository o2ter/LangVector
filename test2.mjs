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
import { LlamaDevice, Similarity, LlamaPoolingType } from './dist/index.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const model = await LlamaDevice.loadModel({
  modelPath: path.join(__dirname, 'models', 'sentence-transformers/all-MiniLM-L6-v2/ggml-model-f16.gguf'),
  useMmap: true,
});

const list = [
  'bonjour!',
  'hello',
  'hi',
  'bye',
  'meta',
  '你好',
  'hello, world',
  'what is your name',
  '你叫咩名',
];

console.log(_.map(list, x => x.slice(0, 6)).join('\t'));

for (const s1 of list) {
  const { vector: v1 } = await model.embedding(s1);
  const result = [];
  for (const s2 of list) {
    const { vector: v2 } = await model.embedding(s2);
    result.push(Similarity.cosine(v1, v2).toFixed(2));
  }
  console.log([...result, s1].join('\t'));
}
