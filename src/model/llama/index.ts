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
import { LLMModel } from '../base';
import { LlamaDevice } from '../../device/llama';
import { SpecialTokenType, DisposedError, LLMTextValue, Vector } from '../../types';
import { LlamaContext } from '../../context/llama';
import { LlamaContextOptions } from '../../context/llama/types';
import { clock } from '../../utils';
import * as llamaCpp from '../../plugins/llamaCpp';
import { LlamaPoolingType } from './types';

export class LlamaModel extends LLMModel<LlamaDevice> {

  /** @internal */
  _model: typeof llamaCpp.LlamaModel;

  /** @internal */
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

  get hasEncoder(): boolean {
    return this._model.hasEncoder();
  }

  get tokens() {
    if (_.isNil(this._model)) throw new DisposedError();
    const _model = this._model;
    return {
      /**
       * The BOS (Beginning Of Sequence) token.
       */
      get bos(): number | undefined {
        const token = _model.tokenBos();
        return token === -1 ? undefined : token;
      },
      /**
       * The EOS (End Of Sequence) token.
       */
      get eos(): number | undefined {
        const token = _model.tokenEos();
        return token === -1 ? undefined : token;
      },
      /**
       * The NL (New Line) token.
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
   * Whether we should prepend a BOS (Beginning Of Sequence) token for evaluations with this model.
   */
  get shouldPrependBosToken(): boolean {
    if (_.isNil(this._model)) throw new DisposedError();
    return !_.isNil(this.tokens.bos) && this._model.shouldPrependBosToken();
  }

  get description(): string {
    return this._model.description();
  }
  /** 
   * The context size the model was trained on
   */
  get contextSize(): number {
    if (_.isNil(this._model)) throw new DisposedError();
    return this._model.contextSize();
  }
  /** 
   * The size of an embedding vector the model can produce
   */
  get embeddingSize(): number {
    if (_.isNil(this._model)) throw new DisposedError();
    return this._model.embeddingSize();
  }
  get totalSize(): number {
    if (_.isNil(this._model)) throw new DisposedError();
    return this._model.totalSize();
  }
  get totalParameters(): number {
    if (_.isNil(this._model)) throw new DisposedError();
    return this._model.totalParameters();
  }
  get vocabularyType() {
    if (_.isNil(this._model)) throw new DisposedError();
    return this._model.vocabularyType();
  }
  get modelSize(): number {
    if (_.isNil(this._model)) throw new DisposedError();
    return this._model.modelSize();
  }

  get chatTemplate() {
    return this.meta['tokenizer.chat_template'];
  }

  get meta() {
    if (_.isNil(this._model)) throw new DisposedError();
    const length = this._model.metaLength();
    const result: Record<string, string | undefined> = {};
    for (let i = 0; i < length; i++) {
      result[this._model.metaKey(i)] = this._model.metaValue(i);
    }
    return result;
  }

  tokenString(token: number): string | undefined {
    if (_.isNil(this._model)) throw new DisposedError();
    return this._model.getTokenString(token);
  }

  tokenAttributes(token: number): number {
    if (_.isNil(this._model)) throw new DisposedError();
    return this._model.getTokenAttributes(token);
  }

  isEogToken(token: number): boolean {
    if (_.isNil(this._model)) throw new DisposedError();
    return this._model.isEogToken(token);
  }

  tokenize(value: LLMTextValue, { addSpecial = false, encodeSpecial = false } = {}): Uint32Array {
    if (_.isNil(this._model)) throw new DisposedError();
    const model = this._model;
    function* _tokenize(value: LLMTextValue): Generator<number, void, undefined> {
      if (_.isNumber(value)) {
        yield value;
      } else if (value instanceof Uint32Array) {
        yield* value;
      } else if (_.isArray(value) && _.every(value, x => _.isNumber(x))) {
        yield* value;
      } else if (_.isString(value)) {
        yield* model.tokenize(value, addSpecial, encodeSpecial);
      } else if (value instanceof SpecialTokenType) {
        yield* model.tokenize(value._text, addSpecial, true);
      } else {
        for (const v of value) yield* _tokenize(v);
      }
    }
    return new Uint32Array([..._tokenize(value)]);
  }
  detokenize(value: LLMTextValue, { removeSpecial = false, decodeSpecial = false } = {}): string {
    if (_.isNil(this._model)) throw new DisposedError();
    const model = this._model;
    function* _detokenize(value: LLMTextValue): Generator<string, void, undefined> {
      if (_.isNumber(value)) {
        yield model.detokenize(new Uint32Array([value]), removeSpecial, decodeSpecial);
      } else if (value instanceof Uint32Array) {
        yield model.detokenize(value, removeSpecial, decodeSpecial);
      } else if (_.isArray(value) && _.every(value, x => _.isNumber(x))) {
        yield model.detokenize(new Uint32Array(value), removeSpecial, decodeSpecial);
      } else if (_.isString(value)) {
        yield value;
      } else if (value instanceof SpecialTokenType) {
        yield value._text;
      } else {
        for (const v of value) yield* _detokenize(v);
      }
    }
    return [..._detokenize(value)].join('');
  }

  applyChatTemplate(msgs: { role: string; content: string; }[]): string | undefined {
    const template = this.chatTemplate;
    return template ? this._model.applyChatTemplate(template, msgs) : undefined;
  }

  createContext(options: LlamaContextOptions = {}) {
    const _options = _.pickBy(options, v => !_.isNil(v));
    const ctx = new llamaCpp.LlamaContext(this._model, _options);
    return new LlamaContext(this, ctx, _options);
  }

  async embedding(value: LLMTextValue, { batchSize, threads, poolingType, normalize }: {
    batchSize?: number;
    threads?: number;
    poolingType?: LlamaPoolingType;
    normalize?: number;
  } = {}) {
    const time = clock();
    const tokens = this.tokenize(value, { addSpecial: true });
    const _batchSize = batchSize ?? tokens.length;
    const ctx = new llamaCpp.LlamaEmbeddingContext(this._model, _.pickBy({
      contextSize: tokens.length,
      batchSize: _batchSize,
      threads,
      poolingType,
    }, v => !_.isNil(v)));
    for (let i = 0; i < tokens.length; i += _batchSize) {
      await ctx.eval(tokens.subarray(i, i + _batchSize), i, i + _batchSize >= tokens.length);
    }
    const vector = ctx.embedding(_.pickBy({ normalize }, v => !_.isNil(v))) as Vector;
    ctx.dispose();
    return { type: 'embedding', vector, time: clock() - time } as const;
  }

}
