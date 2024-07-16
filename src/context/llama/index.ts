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
    return this._ctx.stateSize();
  }
  /**
   * The context size of context.
   */
  get contextSize(): number {
    if (_.isNil(this._ctx)) throw new DisposedError();
    return this._ctx.contextSize();
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
    if (_.isNil(chatWrapper)) return [];
    const history = chatWrapper.generateChatHistory(this, this.tokens);
    this._chat_history = history;
    return history;
  }

  set chatHistory(value) {

    const chatWrapper = this.chatWrapper;
    if (_.isNil(chatWrapper)) return;

    this._worker.sync(async () => {

      const state = await chatWrapper.generateContextState(this, value);
      this._tokens = _.flatMap(state, x => _.isArray(x.tokens) ? x.tokens : [...x.tokens]);

      const _state = await this._contextShiftStrategy();
      await this._updateTokens(_state);
    });
  }

  private _removeTokens(startPos: number, endPos: number): boolean {
    return this._ctx.removeTokens(startPos, endPos);
  }

  private _shiftTokens(startPos: number, endPos: number, shiftDelta: number) {
    return this._ctx.shiftTokens(startPos, endPos, shiftDelta);
  }

  private async _updateTokens(value: Uint32List) {

    const tokens = _.isArray(value) ? value : [...value];

    if (tokens.length > this.contextSize) {
      throw Error('Invalid context shift operation');
    }

    const diff = _.findIndex(this._ctx_state, (x, i) => x !== tokens[i]);

    if (diff !== -1) {

      this._removeTokens(diff, this._ctx_state.length);
      this._ctx_state = this._ctx_state.slice(0, diff);

      const _tokens = tokens.slice(diff);
      await this._ctx.eval(new Uint32Array(_tokens), this._ctx_state.length);
      this._ctx_state.push(..._tokens);

    } else if (this._ctx_state.length < tokens.length) {

      await this._ctx.eval(new Uint32Array(tokens), this._ctx_state.length);
      this._ctx_state.push(...tokens);
    }
  }

  private _grammarEvaluationState(
    grammar: LlamaGrammar,
  ) {
    return new llamaCpp.LlamaGrammarEvaluationState(this._ctx, grammar._grammar);
  }

  private _sampleCandidates(
    options: LLamaChatPromptOptions,
  ) {

    const lastTokens = options.repeatPenalty ? options.repeatPenalty.lastTokens ?? 64 : 0;
    const penalizeNewLine = options.repeatPenalty ? options.repeatPenalty.penalizeNewLine : null;
    const punishTokensFilter = options.repeatPenalty ? options.repeatPenalty.punishTokensFilter : null;

    const repeatPenalty = {
      punishTokens: () => {
        let tokens: Uint32List = this._tokens.slice(-lastTokens);
        tokens = punishTokensFilter ? punishTokensFilter(this, tokens) : tokens;
        if (penalizeNewLine !== false) {
          const nlToken = this.model.tokens.nl;
          if (nlToken != null) tokens = tokens.filter(token => token !== nlToken);
        }
        return tokens;
      },
      ...options.repeatPenalty ? options.repeatPenalty : {},
    };
    const punishTokens = options.repeatPenalty === false ?
      [] : _.isFunction(repeatPenalty.punishTokens) ?
        repeatPenalty.punishTokens(this) :
        repeatPenalty.punishTokens;
    const tokenBias = [
      ..._.isFunction(options.tokenBias) ?
        options.tokenBias(this) :
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

  private async _contextShiftStrategy() {

    const contextShiftStrategy = this._options.chatOptions?.contextShiftStrategy;
    if (contextShiftStrategy) return contextShiftStrategy(this);

    const chatWrapper = this._options.chatOptions?.chatWrapper;
    const maxTokens = Math.floor(this.contextSize * 0.9);

    if (!chatWrapper) {
      const bos = this.model.tokens.bos
      if (bos && _.first(this._tokens) === bos) {
        return [bos, ...this._tokens.slice(-maxTokens)];
      } else {
        return this._tokens.slice(-maxTokens);
      }
    }

    const state = chatWrapper.generateContextState(this, this.chatHistory);

    const sys = _.first(state)?.item.type === 'system' ? _.first(state) : undefined;
    const rest = sys ? _.drop(state, 1) : state;
    const result: Uint32List[] = [];

    let remain = sys ? maxTokens - sys.tokens.length : maxTokens;

    for (const item of _.reverse(rest)) {
      if (remain <= 0) break;
      result.push(item.tokens);
      remain -= item.tokens.length;
    }
    if (sys) result.push(sys.tokens);

    return _.flatMap(_.reverse(result), x => _.isArray(x) ? x : [...x]);
  }

  private async _decodeTokens(value: LLMTextValue) {

    const chatWrapper = this._options.chatOptions?.chatWrapper;

    if (!this._tokens.length && chatWrapper) {

      const state = chatWrapper.generateContextState(this, []);

      const _tokens = _.flatMap(state, x => _.isArray(x.tokens) ? x.tokens : [...x.tokens]);
      this._tokens = _tokens;

      await this._ctx.eval(new Uint32Array(_tokens), this._ctx_state.length);
      this._ctx_state.push(..._tokens);
    }

    const tokens = this.model._tokenize(value);
    this._tokens.push(...tokens);
    this._chat_history = undefined;

    if (this._ctx_state.length + tokens.length > this.contextSize) {
      const _state = await this._contextShiftStrategy();
      await this._updateTokens(_state);
    }

    await this._ctx.eval(tokens, this._ctx_state.length);
    this._ctx_state.push(...tokens);
  }

  private async _evaluate(
    value: LLMTextValue,
    options: LLamaChatPromptOptions,
    onToken: (token: number, time: number) => void,
  ) {

    const grammar = options.grammar ? this._grammarEvaluationState(options.grammar) : null;
    const stopTriggers = _.map(options.stopTriggers, x => this.model._tokenize(x));

    return await this._worker.sync(async () => {

      if (_.isNil(this._ctx)) throw new DisposedError();

      const totalTime = clock();

      await this._decodeTokens(value);

      let maxTokens = options.maxTokens ?? -1;
      while (maxTokens--) {

        if (options.signal?.aborted) return {
          stopReason: 'abort',
          totalTime: clock() - totalTime,
        } as const;

        const time = clock();
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

        if (this.model.isEogToken(sample)) return {
          stopReason: 'eogToken',
          totalTime: clock() - totalTime,
        } as const;

        if (grammar) grammar.acceptToken(sample);
        await this._decodeTokens([sample]);
        onToken(sample, clock() - time);

        for (const trigger of stopTriggers) {
          let offset = this._tokens.length - trigger.length;
          if (offset >= 0 && trigger.every((v, i) => v === this._tokens[i + offset])) return {
            stopReason: 'stopTrigger',
            totalTime: clock() - totalTime,
          } as const;
        }
      }

      return {
        stopReason: 'maxTokens',
        totalTime: clock() - totalTime,
      } as const;
    });
  }

  prompt(value: LLMTextValue, options: LLamaChatPromptOptions = {}) {
    return EventIterator<
      {
        token: number;
        time: number;
      },
      Awaited<ReturnType<LlamaContext['_evaluate']>>
    >(async (push, resolve) => {
      resolve(await this._evaluate(value, options, (token, time) => push({ token, time })));
    });
  }

  evaluate(value: LLMTextValue) {
    return this.prompt(value, { maxTokens: 0 });
  }
}
