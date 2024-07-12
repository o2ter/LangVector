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
import { LLMModel } from './base';
import { LlamaDevice } from '../device/llama';
import * as llamaCpp from '../plugins/llamaCpp';
import { DisposedError } from '../types';

export class LlamaModel extends LLMModel<LlamaDevice> {

  private _model: typeof llamaCpp.LlamaModel;

  constructor(device: LlamaDevice, model: typeof llamaCpp.LlamaModel) {
    super(device);
    this._model = model;
  }

  async dispose() {
    if (_.isNil(this._model)) return;
    this._model.dispose();
    this._model = null;
  }

  get disposed() {
    return _.isNil(this._model);
  }

  get tokens() {
    if (_.isNil(this._model)) throw new DisposedError();
    const _model = this._model;
    return {
      /**
       * @returns The BOS (Beginning Of Sequence) token.
       */
      get bos(): number | undefined {
        const token = _model.tokenBos();
        return token === -1 ? undefined : token;
      },
      /**
       * @returns The EOS (End Of Sequence) token.
       */
      get eos(): number | undefined {
        const token = _model.tokenEos();
        return token === -1 ? undefined : token;
      },
      /**
       * @returns The NL (New Line) token.
       */
      get nl(): number | undefined {
        const token = _model.tokenNl();
        return token === -1 ? undefined : token;
      },
      get prefix(): number | undefined {
        const token = _model.prefixToken();
        return token === -1 ? undefined : token;
      },
      get middle(): number | undefined {
        const token = _model.middleToken();
        return token === -1 ? undefined : token;
      },
      get suffix(): number | undefined {
        const token = _model.suffixToken();
        return token === -1 ? undefined : token;
      },
      get eot(): number | undefined {
        const token = _model.eotToken();
        return token === -1 ? undefined : token;
      },
    };
  }
  /**
   * @returns Whether we should prepend a BOS (Beginning Of Sequence) token for evaluations with this model.
   */
  get shouldPrependBosToken(): boolean {
    if (_.isNil(this._model)) throw new DisposedError();
    return !_.isNil(this.tokens.bos) && this._model.shouldPrependBosToken();
  }

  get description(): string {
    return this._model.getModelDescription();
  }
  /** The context size the model was trained on */
  get trainContextSize(): number {
    if (_.isNil(this._model)) throw new DisposedError();
    return this._model.getTrainContextSize();
  }
  /** The size of an embedding vector the model can produce */
  get embeddingVectorSize(): number {
    if (_.isNil(this._model)) throw new DisposedError();
    return this._model.getEmbeddingVectorSize();
  }
  get totalSize(): number {
    if (_.isNil(this._model)) throw new DisposedError();
    return this._model.getTotalSize();
  }
  get totalParameters(): number {
    if (_.isNil(this._model)) throw new DisposedError();
    return this._model.getTotalParameters();
  }
  get vocabularyType() {
    if (_.isNil(this._model)) throw new DisposedError();
    return this._model.getVocabularyType();
  }
  get modelSize(): number {
    if (_.isNil(this._model)) throw new DisposedError();
    return this._model.getModelSize();
  }
}
