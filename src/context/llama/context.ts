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
import { LlamaModel } from '../../model/llama';
import { DisposedError } from '../../types';
import type { LlamaSession } from '../../session/llama';
import * as llamaCpp from '../../plugins/llamaCpp';
import { Awaitable } from '@o2ter/utils-js';

export class _LlamaContext {

  model: LlamaModel;
  ctx: typeof llamaCpp.LlamaContext;

  seq: LlamaSession[] = [];

  lock = false;
  jobs: (() => Promise<void>)[] = [];

  constructor(model: LlamaModel, context: typeof llamaCpp.LlamaContext) {
    this.model = model;
    this.ctx = context;
  }

  async dispose() {
    return await this._sync(async () => {
      if (_.isNil(this.ctx)) return;
      this.ctx.dispose();
      this.ctx = null;
    });
  }
  get disposed() {
    return _.isNil(this.ctx);
  }

  async _sync<T = void>(callback: () => Awaitable<T>) {
    return await new Promise<T>(async (res, rej) => {
      this.jobs.push(async () => {
        try {
          res(await callback());
        } catch (e) {
          rej(e);
        }
      });

      if (!this.lock) {
        this.lock = true;
        for (const job of this.jobs) {
          await job();
        }
        this.jobs = [];
        this.lock = false;
      }
    });
  }

  get maxSequence(): number {
    if (_.isNil(this.ctx)) throw new DisposedError();
    return this.ctx.maxSequence();
  }

  get contextSize(): number {
    if (_.isNil(this.ctx)) throw new DisposedError();
    return this.ctx.contextSize();
  }

  async disposeSeq(idx: number) {
    return await this._sync(async () => {
      if (_.isNil(this.ctx)) return;
      this.seq = _.filter(this.seq, s => s._idx !== idx);
      this.ctx.disposeSequence(idx);
    });
  }

  availableSeqIdx() {
    const max = this.maxSequence;
    for (let i = 0; i < max; i++) {
      if (_.find(this.seq, s => s._idx === i)) continue;
      return i;
    }
  }
}
