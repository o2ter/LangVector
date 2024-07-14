//
//  types.ts
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

import { Awaitable } from '@o2ter/utils-js';
import { ChatWrapper } from '../../chat/types';
import type { LlamaSession } from './index';

export type LlamaSessionOptions = {

  tokenCompression?: (session: LlamaSession) => Awaitable<Uint32List>;

  chatWrapper?: ChatWrapper;
};

export type LlamaSequenceRepeatPenalty = {
  /** Tokens to lower the predication probability of to be the next predicted token */
  punishTokens: Uint32List | (() => Uint32List);
  /**
   * The relative amount to lower the probability of the tokens in `punishTokens` by
   * Defaults to `1.1`.
   * Set to `1` to disable.
   */
  penalty?: number;
  /**
   * For n time a token is in the `punishTokens` array, lower its probability by `n * frequencyPenalty`
   * Disabled by default (`0`).
   * Set to a value between `0` and `1` to enable.
   */
  frequencyPenalty?: number;
  /**
   * Lower the probability of all the tokens in the `punishTokens` array by `presencePenalty`
   * Disabled by default (`0`).
   * Set to a value between `0` and `1` to enable.
   */
  presencePenalty?: number;
};
