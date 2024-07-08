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

import { LlamaModel } from '../model/llama';
import {
  LlamaModel as _LlamaModel,
  LlamaContext as _LlamaContext,
  LlamaEmbeddingContext as _LlamaEmbeddingContext,
  LlamaContextOptions
} from '../plugins/llama-cpp';
import { LLMContext } from './index';

export class LlamaContext extends LLMContext<LlamaModel> {

  private _ctx?: _LlamaContext;
  private _embedding?: _LlamaEmbeddingContext;

  private _options?: LlamaContextOptions;

  constructor(model: LlamaModel, options?: LlamaContextOptions) {
    super(model);
    this._options = options;
  }

  async dispose() {
    if (this._ctx && !this._ctx.disposed) await this._ctx.dispose();
    if (this._embedding && !this._embedding.disposed) await this._embedding.dispose();
  }

  get disposed() {
    if (this._ctx && !this._ctx.disposed) return false;
    if (this._embedding && !this._embedding.disposed) return false;
    return true;
  }
}
