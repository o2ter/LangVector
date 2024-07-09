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
import { llamaCpp } from '../plugins/llama-cpp';
import { LLMModel } from './base';

export class LlamaModel extends LLMModel<LlamaDevice, llamaCpp.LlamaModel> {

  async dispose() {
    if (!this._model.disposed) await this._model.dispose();
  }

  get disposed() {
    return this._model.disposed;
  }

  async createContext(options?: llamaCpp.LlamaContextOptions) {
    return new LlamaContext(this, options);
  }

  _createContext(options?: llamaCpp.LlamaContextOptions) {
    return this._model.createContext(options);
  }

  _createEmbeddingContext(options?: llamaCpp.LlamaEmbeddingContextOptions) {
    return this._model.createEmbeddingContext(options);
  }

  get tokens() {
    return this._model.tokens;
  }
  get filename() {
    return this._model.filename;
  }
  get fileInfo() {
    return this._model.fileInfo;
  }
  get fileInsights() {
    return this._model.fileInsights;
  }

  get gpuLayers() {
    return this._model.gpuLayers;
  }

  get size() {
    return this._model.size;
  }
  get flashAttentionSupported() {
    return this._model.flashAttentionSupported;
  }
  get defaultContextFlashAttention() {
    return this._model.defaultContextFlashAttention;
  }

  tokenize(text: llamaCpp.BuiltinSpecialTokenValue, specialTokens: 'builtin'): llamaCpp.Token[];
  tokenize(text: string, specialTokens?: boolean, options?: "trimLeadingSpace"): llamaCpp.Token[];
  tokenize(text: string, specialTokens?: boolean | 'builtin', options?: "trimLeadingSpace") {
    if (specialTokens === 'builtin') {
      return this._model.tokenize(text as llamaCpp.BuiltinSpecialTokenValue, specialTokens);
    }
    return this._model.tokenize(text, specialTokens, options);
  }

  detokenize(tokens: readonly llamaCpp.Token[], specialTokens?: boolean) {
    return this._model.detokenize(tokens, specialTokens);
  }
  getTokenAttributes(token: llamaCpp.Token) {
    return this._model.getTokenAttributes(token);
  }

  isSpecialToken(token?: llamaCpp.Token) {
    return this._model.isSpecialToken(token);
  }

  isEogToken(token?: llamaCpp.Token) {
    return this._model.isEogToken(token);
  }
}
