//
//  index.ts
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
import { LLMSession } from '../base';
import { LlamaContext } from '../../context/llama';
import { LlamaDevice } from '../../device/llama';
import { LlamaModel } from '../../model/llama';
import { _LlamaContext } from '../../context/llama/context';
import { LlamaSessionOptions } from './types';
import { DisposedError } from '../../types';

export class LlamaSession extends LLMSession<LlamaDevice, LlamaModel, LlamaContext> {

  /** @internal */
  _options: LlamaSessionOptions;
  /** @internal */
  _idx: number;
  /** @internal */
  _ctx: _LlamaContext;
  /** @internal */
  _disposed = false;

  /** @internal */
  constructor(
    pool: LlamaContext,
    ctx: _LlamaContext,
    idx: number,
    options: LlamaSessionOptions
  ) {
    super(pool);
    this._options = options;
    this._idx = idx;
    this._ctx = ctx;
  }

  async dispose() {
    if (this._disposed) return;
    this._ctx.disposeSeq(this._idx);
    this._disposed = true;
  }
  get disposed() {
    return this._disposed;
  }

  /**
   * The context size of context.
   */
  get contextSize() {
    if (this._disposed) throw new DisposedError();
    return this._ctx.contextSize;
  }

}