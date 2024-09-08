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
import { myers } from 'myers.js';
import { clock } from '../../utils';
import { Worker } from './worker';
import { Awaitable, EventIterator } from '@o2ter/utils-js';
import { LLMContext } from '../base';
import { LlamaModel } from '../../model/llama';
import { LlamaDevice } from '../../device/llama';
import { LLamaChatPromptOptions, LlamaContextOptions } from './types';
import { DisposedError, LLMTextValue } from '../../types';
import { ChatHistoryItem } from '../../chat/wrapper/types';
import * as llamaCpp from '../../plugins/llamaCpp';

export class LlamaContext extends LLMContext<LlamaModel> {

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
   * The max context size of context.
   */
  get maxContextSize(): number {
    if (_.isNil(this._ctx)) throw new DisposedError();
    return this._ctx.contextSize();
  }
  /**
   * The context size of context.
   */
  get contextSize(): number {
    if (_.isNil(this._ctx)) throw new DisposedError();
    return this._ctx_state.length;
  }
  /**
   * The batch size of context.
   */
  get batchSize(): number {
    if (_.isNil(this._ctx)) throw new DisposedError();
    return this._ctx.batchSize();
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
    const history = chatWrapper.decodeChatHistory(this, this.tokens);
    this._chat_history = history;
    return history;
  }

  set chatHistory(value) {

    const chatWrapper = this.chatWrapper;
    if (_.isNil(chatWrapper)) return;

    this._worker.sync(async () => {

      this._chat_history = undefined;

      const state = await chatWrapper.encodeContextState(this, value);
      this._tokens = _.flatMap(state, x => _.isArray(x.tokens) ? x.tokens : [...x.tokens]);

      const _state = await this._contextShiftStrategy();
      await this._updateTokens(_state);
    });
  }

  /** @internal */
  private _removeTokens(startPos: number, endPos: number): boolean {
    return this._ctx.removeTokens(startPos, endPos);
  }

  /** @internal */
  private _shiftTokens(startPos: number, endPos: number, shiftDelta: number) {
    return this._ctx.shiftTokens(startPos, endPos, shiftDelta);
  }

  /** @internal */
  private async _updateTokens(value: Uint32List) {

    const tokens = _.isArray(value) ? value : [...value];

    if (tokens.length > this.maxContextSize) {
      throw Error('Invalid context shift operation');
    }

    let pos = 0;
    for (const diff of await myers(this._ctx_state, tokens)) {
      if (diff.equivalent) pos += diff.equivalent.length;
      if (diff.remove) {
        this._removeTokens(pos, pos + diff.remove.length);
      }
      if (diff.insert) {
        this._shiftTokens(pos, -1, diff.insert.length);
        await this._eval(new Uint32Array(diff.insert), pos);
        pos += diff.insert.length;
      }
    }

    this._ctx_state = tokens;
  }

  /** @internal */
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
      tokenBias: _.map(tokenBias, ([key, value]) => ({
        key: key,
        value: value === 'never' ? Number.NEGATIVE_INFINITY : value,
      })),
      repeatPenalty: repeatPenalty.penalty,
      repeatPenaltyPresencePenalty: repeatPenalty.presencePenalty,
      repeatPenaltyFrequencyPenalty: repeatPenalty.frequencyPenalty,
      repeatPenaltyTokens: punishTokens instanceof Uint32Array ? punishTokens : new Uint32Array(punishTokens),
    }, v => !_.isNil(v)));
  }

  /** @internal */
  private async _contextShiftStrategy() {

    const contextShiftStrategy = this._options.chatOptions?.contextShiftStrategy;
    if (contextShiftStrategy) return contextShiftStrategy(this);

    const chatWrapper = this._options.chatOptions?.chatWrapper;
    const maxTokens = Math.floor(this.maxContextSize * 0.9);

    if (!chatWrapper) {
      const bos = this.model.tokens.bos
      if (bos && _.first(this._tokens) === bos) {
        return [bos, ...this._tokens.slice(-maxTokens)];
      } else {
        return this._tokens.slice(-maxTokens);
      }
    }

    const state = chatWrapper.encodeContextState(this, this.chatHistory);

    const sys = _.first(state)?.item.type === 'system' ? _.first(state) : undefined;
    const history = sys ? _.drop(state, 1) : state;
    const result: Uint32List[] = [];

    let remain = sys ? maxTokens - sys.tokens.length : maxTokens;

    for (const item of _.reverse(history)) {
      if (remain <= 0) break;
      result.push(item.tokens);
      remain -= item.tokens.length;
    }
    if (sys) result.push(sys.tokens);

    return _.flatMap(_.reverse(result), x => _.isArray(x) ? x : [...x]);
  }

  /** @internal */
  private async _eval(tokens: Uint32List, startPos: number) {
    const _tokens = tokens instanceof Uint32Array ? tokens : new Uint32Array(tokens);
    const batchSize = this.batchSize;
    for (let i = 0; i < tokens.length; i += batchSize) {
      await this._ctx.eval(_tokens.subarray(i, i + batchSize), i + startPos, i + batchSize >= _tokens.length);
    }
  }

  /** @internal */
  private async _decodeTokens(value: LLMTextValue) {

    const tokens = this.model.tokenize(value);
    this._tokens.push(...tokens);

    if (this._ctx_state.length + tokens.length > this.maxContextSize) {
      const _state = await this._contextShiftStrategy();
      await this._updateTokens(_state);
    }

    await this._eval(tokens, this._ctx_state.length);
    this._ctx_state.push(...tokens);
  }

  /** @internal */
  private _evaluate_modules() {

    const modules: {
      beginTrigger: string;
      grammar: string;
      stopGenerationTriggers: Uint32List[];
      handle: (tokens: number[]) => Awaitable<LLMTextValue[]>;
    }[] = [];

    const chatWrapper = this._options.chatOptions?.chatWrapper;
    const functions = this._options.chatOptions?.functions;
    const functionGrammar = chatWrapper?.generateFunctionGrammar?.(this);
    if (chatWrapper && functions && functionGrammar) {
      modules.push({
        beginTrigger: functionGrammar.beginTrigger,
        grammar: functionGrammar.grammar,
        stopGenerationTriggers: functionGrammar.stopGenerationTriggers,
        handle: async (tokens) => {
          const calls = chatWrapper.decodeFunctionCalls(this, this.model.detokenize(tokens));
          const results = await Promise.all(_.map(calls, ({ name, params }) => functions[name]?.handler(params)));
          return _.map(results, v => chatWrapper.encodeNextContextState(
            this, functionGrammar.responseRole, functionGrammar.responseEncoder(v)
          ))
        },
      });
    }

    return modules;
  }

  /** @internal */
  private async _evaluate(
    value: LLMTextValue,
    options: LLamaChatPromptOptions,
    onToken: (token: number, time: number) => void,
  ) {

    const totalTime = clock();

    const modules = this._evaluate_modules();
    const chatWrapper = this._options.chatOptions?.chatWrapper;
    const grammar = options.grammar;
    const stopTriggers = _.map(
      options.stopTriggers ?? chatWrapper?.stopGenerationTriggers(this),
      x => this.model.tokenize(x)
    );

    let inputs: LLMTextValue[] = [
      chatWrapper ? chatWrapper.encodeNextContextState(this, 'user', value) : value,
    ];

    return await this._worker.sync(async () => {

      if (_.isNil(this._ctx)) throw new DisposedError();

      try {

        while (inputs.length) {

          await this._decodeTokens(inputs);
          inputs = [];

          let maxTokens = options.maxTokens ?? -1;
          let _grammar = grammar;
          let _modules: typeof modules = [];
          let _selected_module: typeof modules[number] | undefined;
          let _module_records: [number, number][] | undefined;

          loop: while (maxTokens--) {

            if (options.signal?.aborted) return {
              stopReason: 'abort',
              totalTime: clock() - totalTime,
            } as const;

            const time = clock();
            let candidates = this._sampleCandidates(options);

            if (_grammar) {
              _grammar.sampleToken(candidates);
              if (!candidates.isValid()) {
                // logit biases caused grammar sampling to fail, so sampling again without logit biases
                candidates = this._sampleCandidates(_.omit(options, 'tokenBias'));
                _grammar.sampleToken(candidates);
              }
            }

            const sample = await this._ctx.sampleToken(candidates, _.pickBy({
              temperature: options.temperature,
              minP: options.minP,
              topK: options.topK,
              topP: options.topP,
            }, v => !_.isNil(v)));

            if (this.model.isEogToken(sample)) {
              if (_selected_module && !_.isNil(_module_records)) {
                inputs = await _selected_module.handle(_.map(_module_records, ([x]) => x));
                if (!_.isEmpty(inputs)) break loop;
              }
              return {
                stopReason: 'eogToken',
                totalTime: clock() - totalTime,
              } as const;
            }

            for (const trigger of _selected_module?.stopGenerationTriggers ?? stopTriggers) {
              let offset = this._tokens.length - trigger.length;
              if (offset >= 0 && trigger.every((v, i) => v === this._tokens[i + offset])) {
                if (_selected_module && !_.isNil(_module_records)) {
                  inputs = await _selected_module.handle(_.map(_module_records, ([x]) => x));
                  if (!_.isEmpty(inputs)) break loop;
                }
                return {
                  stopReason: 'stopTrigger',
                  totalTime: clock() - totalTime,
                } as const;
              }
            }

            await this._decodeTokens(sample);
            const _time = clock() - time;

            let _record_pushed = false;

            if (_.isNil(_grammar) && _.isNil(_selected_module) && _.isEmpty(_modules)) {
              let str_0 = '';
              let str_1 = this.model.detokenize(sample);
              while (!_.isEmpty(str_1)) {
                for (const module of modules) {
                  if (module.beginTrigger.length > str_1.length) {
                    if (_.startsWith(module.beginTrigger, str_1)) _modules.push(module);
                  } else {
                    if (_.startsWith(str_1, module.beginTrigger)) _modules.push(module);
                  }
                }
                if (_.isEmpty(_modules)) {
                  str_0 += str_1[0];
                  str_1 = str_1.slice(1);
                } else {
                  if (!_.isEmpty(str_0)) {
                    for (const token of this.model.tokenize(str_0)) onToken(token, _time);
                  }
                  _module_records = _.map(this.model.tokenize(str_1), x => [x, _time]);
                  _record_pushed = true;
                  break;
                }
              }
            }

            if (!_.isNil(_module_records)) {
              if (!_record_pushed) _module_records.push([sample, _time]);
            } else {
              onToken(sample, _time);
            }
            if (_grammar) _grammar.acceptToken(sample);

            if (_.isNil(_selected_module) && !_.isEmpty(_modules) && !_.isNil(_module_records)) {
              const record = this.model.detokenize(_.map(_module_records, ([x]) => x));
              for (const module of _modules) {
                if (_.startsWith(record, module.beginTrigger)) {
                  _selected_module = module;
                  _grammar = this._grammarEvaluationState(_selected_module.grammar);
                  for (const [token] of _module_records) _grammar.acceptToken(token);
                  continue loop;
                } else if (record.length >= module.beginTrigger.length) {
                  _modules = _.filter(_modules, x => x !== module);
                }
              }
              if (_.isEmpty(_modules)) {
                for (const [sample, time] of _module_records) onToken(sample, time);
                _grammar = null;
                _selected_module = undefined;
                _module_records = undefined;
              }
            }
          }
        }

        return {
          stopReason: 'maxTokens',
          totalTime: clock() - totalTime,
        } as const;

      } finally {
        this._chat_history = undefined;
      }
    });
  }

  /** @internal */
  private _evaluate_iterator(value: LLMTextValue, options: LLamaChatPromptOptions) {
    type Result = Awaited<ReturnType<LlamaContext['_evaluate']>>;
    return EventIterator(async (push, resolve) => {
      resolve(await this._evaluate(value, options, (token, time) => push({ token, time })));
    }) as AsyncGenerator<{ token: number; time: number; }, { [K in keyof Result]: Result[K] }>;
  }

  prompt(value: LLMTextValue, options: LLamaChatPromptOptions = {}) {
    const iterator = this._evaluate_iterator(value, options);
    return (async function* () {
      const response: number[] = [];
      while (true) {
        const { value, done } = await iterator.next();
        if (done) {
          yield { done: true, response: new Uint32Array(response), ...value } as const;
          return;
        } else {
          response.push(value.token);
          yield { done: false, response: new Uint32Array(response), ...value } as const;
        }
      }
    })();
  }

  async evaluate(value: LLMTextValue) {
    const iterator = this._evaluate_iterator(value, { maxTokens: 0 });
    while (true) {
      const { value, done } = await iterator.next();
      if (done) return value.totalTime;
    }
  }
}
