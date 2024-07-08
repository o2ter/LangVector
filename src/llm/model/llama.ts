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
import { LlamaContext } from '../context/llama';
import { LlamaDevice } from '../device/llama';
import {
  LlamaModel as _LlamaModel,
  BuiltinSpecialTokenValue,
  LlamaContextOptions,
  LlamaEmbeddingContextOptions,
  Token,
} from '../plugins/llama-cpp';
import { LLMModel } from './index';

export class LlamaModel extends LLMModel<LlamaDevice, _LlamaModel> {

  async dispose() {
    if (!this._model.disposed) await this._model.dispose();
  }

  get disposed() {
    return this._model.disposed;
  }

  async createContext(options?: LlamaContextOptions) {
    return new LlamaContext(this, options);
  }

  _createContext(options?: LlamaContextOptions) {
    return this._model.createContext(options)
  }

  _createEmbeddingContext(options?: LlamaEmbeddingContextOptions) {
    return this._model.createEmbeddingContext(options)
  }

  tokenize(text: BuiltinSpecialTokenValue, specialTokens: 'builtin'): Token[];
  tokenize(text: string, specialTokens?: boolean, options?: "trimLeadingSpace"): Token[];
  tokenize(text: string, specialTokens?: boolean | 'builtin', options?: "trimLeadingSpace") {
    if (specialTokens === 'builtin') {
      return this._model.tokenize(text as BuiltinSpecialTokenValue, specialTokens);
    }
    return this._model.tokenize(text, specialTokens, options);
  }

  detokenize(tokens: readonly Token[], specialTokens?: boolean) {
    return this._model.detokenize(tokens, specialTokens);
  }
  getTokenAttributes(token: Token) {
    return this._model.getTokenAttributes(token);
  }

  isSpecialToken(token?: Token) {
    return this._model.isSpecialToken(token);
  }

  isEogToken(token?: Token) {
    return this._model.isEogToken(token);
  }
}
