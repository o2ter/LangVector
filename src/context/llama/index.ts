//
//  llama.ts
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
import { LLMContext } from '../base';
import { LlamaModel } from '../../model/llama';
import { LlamaDevice } from '../../device/llama';
import { LlamaContextOptions } from './types';
import { _LlamaContext } from './context';

export class LlamaContext extends LLMContext<LlamaDevice, LlamaModel> {

  private _options: LlamaContextOptions;
  private _pool: _LlamaContext[] = [];

  constructor(model: LlamaModel, options: LlamaContextOptions) {
    super(model);
    this._options = options;
  }

  async dispose() {
    for (const ctx of this._pool) {
      ctx.dispose();
    }
    this._pool = [];
  }

  get disposed() {
    return _.isEmpty(this._pool);
  }

  private get _available_context() {
    for (const ctx of this._pool) {
      if (ctx.seq.length < ctx.maxSequence) return ctx;
    }
    const ctx = new _LlamaContext(this.model, this._options);
    this._pool.push(ctx);
    return ctx;
  }

  /**
   * The context size of context.
   */
  get contextSize(): number {
    return _.first(this._pool)?.contextSize ?? 0;
  }
}
