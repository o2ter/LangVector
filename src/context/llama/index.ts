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
import { EventIterator } from '@o2ter/utils-js';
import { LLMContext } from '../base';
import { LlamaModel } from '../../model/llama';
import { LlamaDevice } from '../../device/llama';
import { LLamaChatPromptOptions, LlamaContextOptions, LlamaSequenceRepeatPenalty } from './types';
import { DisposedError, LLMTextValue } from '../../types';
import { Worker } from './worker';
import { clock } from '../../utils';
import { ChatHistoryItem } from '../../chat/types';
import { LlamaGrammar } from '../../device/llama/grammar';
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
  _chat_history?: ChatHistoryItem[];
  /** @internal */
  _ctx_state?: number[];

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

  get tokens() {
    return new Uint32Array(this._tokens);
  }

  get chatWrapper() {
    return this._options.chatOptions?.chatWrapper;
  }

  get chatOptions() {
    return this._options.chatOptions;
  }

  get chatHistory() {
    if (!_.isNil(this._chat_history)) return this._chat_history;
    const chatWrapper = this.chatWrapper;
    if (_.isNil(chatWrapper)) return;
    const history = chatWrapper.generateChatHistory(this, this.tokens);
    this._chat_history = history;
    return history;
  }

  /** @internal */
  _removeTokens(startPos: number, endPos: number): boolean {
    return this._ctx.removeTokens(startPos, endPos);
  }

  /** @internal */
  _shiftTokens(startPos: number, endPos: number, shiftDelta: number) {
    return this._ctx.shiftTokens(startPos, endPos, shiftDelta);
  }

  private _grammarEvaluationState(
    grammar: LlamaGrammar,
  ) {
    return new llamaCpp.LlamaGrammarEvaluationState(this._ctx, grammar._grammar);
  }

  private _sampleCandidates(
    options: LLamaChatPromptOptions,
  ) {

    const repeatPenalty = {
      punishTokens: () => {
        return new Uint32Array;
      },
      ...options.repeatPenalty ? options.repeatPenalty : {},
    };
    const punishTokens = options.repeatPenalty === false ?
      [] : _.isFunction(repeatPenalty.punishTokens) ?
        repeatPenalty.punishTokens() :
        repeatPenalty.punishTokens;
    const tokenBias = [
      ..._.isFunction(options.tokenBias) ?
        options.tokenBias() :
        options.tokenBias ?? []
    ];

    return new llamaCpp.LlamaContextSampleCandidates(this._ctx, _.pickBy({
      tokenBiasKeys: new Uint32Array(_.map(tokenBias, x => x[0])),
      tokenBiasValues: new Float32Array(_.map(tokenBias, x => x[1] === 'never' ? Number.NEGATIVE_INFINITY : x[1])),
      repeatPenalty: repeatPenalty.penalty,
      repeatPenaltyPresencePenalty: repeatPenalty.presencePenalty,
      repeatPenaltyFrequencyPenalty: repeatPenalty.frequencyPenalty,
      repeatPenaltyTokens: _.isArrayBuffer(punishTokens) ? punishTokens : new Uint32Array(punishTokens),
    }, v => !_.isNil(v)));
  }

  private async _evaluate(
    value: LLMTextValue,
    options: LLamaChatPromptOptions,
    onToken: (tokens: Uint32Array) => void,
  ): Promise<number> {

    const tokens = this.model._tokenize(value);
    const grammar = options.grammar ? this._grammarEvaluationState(options.grammar) : null;

    return await this._worker.sync(async () => {

      if (_.isNil(this._ctx)) throw new DisposedError();

      this._tokens.push(...tokens);

      const time = clock();

      await this._ctx.eval(tokens);

      if (options.maxTokens === 0) return clock() - time;

      let maxTokens = options.maxTokens ?? -1;

      while (maxTokens--) {

        if (options.signal?.aborted) return clock() - time;

        let candidates = this._sampleCandidates(options);

        if (grammar) {
          grammar.sampleToken(candidates);
          if (!candidates.isValid()) {
            // logit biases caused grammar sampling to fail, so sampling again without logit biases
            candidates = this._sampleCandidates(_.omit(options, 'tokenBias'));
            grammar.sampleToken(candidates);
          }
        }

        const sample = await this._ctx.sampleToken(candidates, _.pickBy({
          temperature: options.temperature,
          minP: options.minP,
          topK: options.topK,
          topP: options.topP,
        }, v => !_.isNil(v)));

        if (grammar && !this.model.isEogToken(sample)) {
          grammar.acceptToken(sample);
        }

        onToken(sample);
      }

      return clock() - time;
    });
  }

  prompt(value: LLMTextValue, options: LLamaChatPromptOptions = {}) {
    return EventIterator<Uint32Array, Awaited<ReturnType<LlamaContext['_evaluate']>>>(async (push, resolve) => {
      resolve(await this._evaluate(value, options, push));
    });
  }

  evaluate(value: LLMTextValue) {
    return this.prompt(value, { maxTokens: 0 });
  }
}
