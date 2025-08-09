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

const embedding = (model, value) => Proto.run('llm_embedding', { model_name: model, value })

export const Similarity = () => {

  const [model, setModel] = useState('sentence-transformers/all-MiniLM-L6-v2/ggml-model-f16.gguf');
  const [method, setMethod] = useState('cosine');

  const [source, setSource] = useState('That is a happy person');
  const [compare, setCompare] = useState([
    'That is a happy dog',
    'That is a very happy person',
    'Today is a sunny day',
  ]);

  const { resource: models } = useResource(() => Proto.run('llm_models'), []);

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
    <div className='d-flex flex-column flex-fill'>
      <Container>
        <span className='mt-2'>Model</span>
        <div className='d-flex flex-row flex-fill'>
          <Select
            value={model}
            options={_.map(models, x => ({ value: x, label: x }))}
            onValueChange={v => setModel(v)}
          />
          <SegmentedControl
            value={method}
            segments={[
              { label: 'cosine', value: 'cosine' },
              { label: 'distance', value: 'distance' },
            ]}
            onChange={v => setMethod(v)}
          />
        </div>
        <span className='mt-2'>Source Sentence</span>
        <TextInput value={source} onChangeText={setSource} />
        <span className='mt-2'>Sentences to compare to</span>
        {_.map([...compare, ''], (x, i) => (
          <div key={i} className='d-flex flex-row flex-fill mt-1'>
            <TextInput
              classes='flex-fill'
              value={x}
              onChangeText={s => setCompare(v => {
                const a = [...v];
                a[i] = s;
                return a;
              })}
            />
            {result?.[i] && (
              <>
                <span className='ml-2'>time: {result[i].time}</span>
                <span className='ml-2'>{method}: {result[i][method]}</span>
              </>
            )}
          </div>
        ))}
      </Container>
    </div>
  );
};
