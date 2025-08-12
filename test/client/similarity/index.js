//
//  index.js
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
import { useResource, useState } from 'frosty';
import { Similarity as _Similarity } from '../../../src/similarity';

const embedding = async (model, value) => {
  const res = await fetch('/api/llm_embedding', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({ model_name: model, value }),
  });
  return res.json();
}

export const Similarity = () => {

  const [model, setModel] = useState('sentence-transformers/all-MiniLM-L6-v2/ggml-model-f16.gguf');

  const [source, setSource] = useState('That is a happy person');
  const [compare, setCompare] = useState([
    'That is a happy dog',
    'That is a very happy person',
    'Today is a sunny day',
  ]);

  const { resource: models } = useResource(async () => {
    const res = await fetch('/api/llm_models');
    return res.json();
  }, []);

  const { resource: result } = useResource({
    fetch: async () => {
      if (!model) return;
      const { vector: s } = await embedding(model, source);
      const c = await Promise.all(_.map(compare, x => embedding(model, x)));
      return _.map(c, ({ vector: v, time }) => ({
        time,
        distance: _Similarity.distance(s, v),
        cosine: _Similarity.cosine(s, v),
      }));
    },
    debounce: { wait: 1000 },
  }, [model, source, compare]);

  return (
    <div className='d-flex flex-column h-100'>
      <span className='mt-2'>Model</span>
      <select
        className='form-control'
        value={model}
        onChange={e => setModel(e.currentTarget.value)}
      >
        {_.map(models, x => (
          <option value={x}>{x}</option>
        ))}
      </select>
      <span className='mt-2'>Source Sentence</span>
      <input className='form-control' value={source} onChange={e => setSource(e.currentTarget.value)} />
      <span className='mt-2'>Sentences to compare to</span>
      {_.map([...compare, ''], (x, i) => (
        <div key={i} className='d-flex flex-row mt-1 gap-3'>
          <input
            className='flex-fill form-control'
            value={x}
            onChange={e => setCompare(v => {
              const a = [...v];
              a[i] = e.currentTarget.value;
              return a;
            })}
          />
          {result?.[i] && (
            <>
              <span className='ml-2'>time: {result[i].time}</span>
              <span className='ml-2'>cosine: {result[i].cosine}</span>
              <span className='ml-2'>distance: {result[i].distance}</span>
            </>
          )}
        </div>
      ))}
    </div>
  );
};
