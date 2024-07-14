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
import { LLamaChatPromptOptions, LlamaContextOptions } from './types';
import { DisposedError, LLMTextValue } from '../../types';
import { Worker } from './worker';
import { clock } from '../../utils';
import * as llamaCpp from '../../plugins/llamaCpp';

export class LlamaContext extends LLMContext<LlamaDevice, LlamaModel> {

  /** @internal */
  _ctx: typeof llamaCpp.LlamaContext;
  /** @internal */
  _options: LlamaContextOptions;

  /** @internal */
  _worker = new Worker;

  /** @internal */
  _tokens: number[] = [];
  /** @internal */
  _ctx_state: number[] = [];

  /** @internal */
  constructor(model: LlamaModel, ctx: typeof llamaCpp.LlamaContext, options: LlamaContextOptions) {
    super(model);
    this._ctx = ctx;
    this._options = options;
  }

  async dispose() {
    return await this._worker.sync(async () => {
      if (_.isNil(this._ctx)) return;
      this._ctx.dispose();
      this._ctx = null;
    });
  }

  get disposed() {
    return _.isNil(this._ctx);
  }

  /**
   * The state size of context.
   */
  get stateSize(): number {
    if (_.isNil(this._ctx)) throw new DisposedError();
    return this._ctx.stateSize;
  }
  /**
   * The context size of context.
   */
  get contextSize(): number {
    if (_.isNil(this._ctx)) throw new DisposedError();
    return this._ctx.contextSize;
  }

  private async _evaluate(
    value: LLMTextValue,
    options: LLamaChatPromptOptions = {},
  ): Promise<number> {

    const tokens = this.model._tokenize(value);

    return await this._worker.sync(async () => {

      if (_.isNil(this._ctx)) throw new DisposedError();

      this._tokens.push(...tokens);

      const time = clock();

      await this._ctx.ctx.eval(tokens);

      return clock() - time;
    });
  }

  evaluate(value: LLMTextValue) {
    return this._evaluate(value);
  }

  prompt(value: LLMTextValue, options: LLamaChatPromptOptions = {}) {
    return this._evaluate(value, options);
  }
}
