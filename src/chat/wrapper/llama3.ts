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

import _ from 'lodash';
import { ChatHistoryItem, ChatModelFunctionCall, ChatSystemMessage, ChatWrapper } from './../types';
import { LlamaContext } from '../../context/llama';
import { LLMTextValue, SpecialToken } from '../../types';
import { tokenEndsWith, tokenFind, tokenStartsWith } from '../../utils';

const functionCallPrefix = '||call: ';
const eotToken = SpecialToken('<|eot_id|>');
const beginOfTextToken = SpecialToken('<|begin_of_text|>');
const endOfTextToken = SpecialToken('<|end_of_text|>');
const startHeaderToken = SpecialToken('<|start_header_id|>');
const endHeaderToken = SpecialToken('<|end_header_id|>');

const roleHeader = (role: string) => [startHeaderToken, role, endHeaderToken, '\n\n'];

export class Llama3ChatWrapper implements ChatWrapper {

  stopGenerationTriggers(ctx: LlamaContext) {
    const { eot, eos } = ctx.model.tokens;
    return _.filter([
      _.compact([eot]),
      _.compact([eos]),
      eotToken,
      endOfTextToken,
    ], x => !_.isEmpty(x));
  }

  generateSystemMessage(ctx: LlamaContext) {
    return [
      'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible.',
      'If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.',
      'If you don\'t know the answer to a question, please don\'t share false information.',
    ].join('\n');
  }

  generateAvailableFunctionsSystemText(ctx: LlamaContext) {
    if (_.isEmpty(ctx.chatOptions?.functions)) return '';
    return [
      'The assistant calls the provided functions as needed to retrieve information instead of relying on existing knowledge.',
      'To fulfill a request, the assistant calls relevant functions in advance when needed before responding to the request, and does not tell the user prior to calling a function.',
      'Provided functions:',
      '```typescript',

      '```',
      '',
      'Calling any of the provided functions can be done like this:',
      '||call: getSomeInfo({ someKey: \"someValue\" })',
      '',
      'Note that the ||call: prefix is mandatory',
      'The assistant does not inform the user about using functions and does not explain anything before calling a function.',
      'After calling a function, the raw result appears afterwards and is not part of the conversation',
      'To make information be part of the conversation, the assistant paraphrases and repeats the information without the function syntax.',
    ].join('\n');
  }

  _defaultSystemMessage(ctx: LlamaContext) {
    return _.compact([this.generateSystemMessage(ctx), this.generateAvailableFunctionsSystemText(ctx)]).join('\n\n');
  }

  encodeNextContextState(ctx: LlamaContext, role: string, value: LLMTextValue) {
    const eot = ctx.model.tokenize(eotToken);
    return _.compact([
      _.isEmpty(ctx._tokens) && [
        beginOfTextToken,
        roleHeader('system'),
        this._defaultSystemMessage(ctx),
        eotToken,
      ],
      !_.isEmpty(ctx._tokens) && !tokenEndsWith(ctx.tokens, eot) && eot,
      roleHeader(role),
      value,
      eotToken,
      roleHeader('assistant'),
    ]);
  }

  encodeContextState(ctx: LlamaContext, chatHistory: ChatHistoryItem[]) {

    const sys = _.first(chatHistory)?.type === 'system' ? _.first(chatHistory) as ChatSystemMessage : undefined;
    const history = sys ? _.drop(chatHistory, 1) : chatHistory;

    const sysMsg = sys?.text ?? this._defaultSystemMessage(ctx);

    const result: {
      item: ChatHistoryItem;
      tokens: Uint32List;
    }[] = [{
      item: {
        type: 'system',
        text: sysMsg,
      },
      tokens: ctx.model.tokenize([
        beginOfTextToken,
        roleHeader('system'),
        sysMsg,
        eotToken,
      ]),
    }];

    for (const item of history) {
      switch (item.type) {
        case 'user':
          result.push({
            item,
            tokens: ctx.model.tokenize([
              roleHeader('user'),
              item.text,
              eotToken,
            ]),
          });
          break;
        case 'model':
          result.push({
            item,
            tokens: ctx.model.tokenize(_.flatMap(item.response, response => {
              if (_.isString(response)) {
                return [
                  roleHeader('assistant'),
                  response,
                  eotToken,
                ];
              }
              return [
                roleHeader('assistant'),
                `${functionCallPrefix}${response.name}(${JSON.stringify(response.params)})`,
                eotToken,
                roleHeader('function_call_result'),
                JSON.stringify(response.result),
                eotToken,
              ];
            })),
          });
          break;
        default: break;
      }
    }

    return result;
  }
  decodeChatHistory(ctx: LlamaContext, tokens: Uint32Array): ChatHistoryItem[] {

    const newline = ctx.model.tokenize('\n\n');
    const beginOfText = ctx.model.tokenize(beginOfTextToken);
    const startHeaderId = ctx.model.tokenize(startHeaderToken);
    const endHeaderId = ctx.model.tokenize(endHeaderToken);
    const eotId = ctx.model.tokenize(eotToken);

    if (!tokenStartsWith(tokens, beginOfText)) throw Error('Invalid chat history');
    tokens = tokens.subarray(beginOfText.length);

    const result: ChatHistoryItem[] = [];

    while (tokens.length > 0) {

      if (!tokenStartsWith(tokens, startHeaderId)) throw Error('Invalid chat history');
      tokens = tokens.subarray(startHeaderId.length);

      const endHeaderIdx = tokenFind(tokens, endHeaderId);
      if (endHeaderIdx === -1) throw Error('Invalid chat history');

      const type = tokens.subarray(0, endHeaderIdx);
      tokens = tokens.subarray(endHeaderIdx + endHeaderId.length);

      while (tokenStartsWith(tokens, newline)) {
        tokens = tokens.subarray(newline.length);
      }

      const _end = tokenFind(tokens, eotId);
      const content = _end === -1 ? tokens : tokens.subarray(0, _end);
      tokens = _end === -1 ? new Uint32Array : tokens.subarray(_end + eotId.length);

      switch (ctx.model.detokenize(type)) {
        case 'system':
          result.push({
            type: 'system',
            text: content,
          });
          break;
        case 'user':
          result.push({
            type: 'user',
            text: ctx.model.detokenize(content),
          });
          break;
        case 'assistant':
          {
            const last = _.last(result);
            if (last?.type === 'model') {
              last.response.push(ctx.model.detokenize(content));
            } else {
              result.push({
                type: 'model',
                response: [ctx.model.detokenize(content)],
              });
            }
          }
          break;
        case 'function_call_result':
          {
            const functions = ctx.chatOptions?.functions;
            if (!functions) throw Error('Invalid chat history');

            const last = _.last(result);
            if (last?.type !== 'model') throw Error('Invalid chat history');

            const calls = _.flatMap(
              _.filter(last.response, x => _.isString(x) && _.startsWith(x, functionCallPrefix)) as string[],
              x => this.decodeFunctionCalls(ctx, x),
            );
            const results = _.filter(last.response, x => !_.isString(x) && x.type === 'functionCall') as ChatModelFunctionCall[];

            if (calls.length <= results.length) throw Error('Invalid chat history');
            const current = calls[results.length];

            result.push({
              type: 'model',
              response: [{
                type: 'functionCall',
                result: JSON.parse(ctx.model.detokenize(content)),
                description: functions[current.name]?.description,
                ...current,
              }],
            });
          }
          break;
        default: throw Error('Invalid chat history');
      }
    }

    return result;
  }

  decodeFunctionCalls(ctx: LlamaContext, tokens: string) {

    const functions = ctx.chatOptions?.functions;
    if (!functions) return [];

    const result: {
      name: string;
      params: any;
    }[] = [];

    for (const line of tokens.split('\n')) {
      const str = line.trim();
      if (_.isEmpty(str)) continue;
      if (!_.startsWith(str, functionCallPrefix)) throw Error('Invalid function call format');
      if (!_.endsWith(str, ')')) throw Error('Invalid function call format');
      for (const name of _.keys(functions)) {
        const prefix = `${functionCallPrefix}${name}(`;
        if (!_.startsWith(str, prefix)) continue;
        result.push({
          name,
          params: JSON.parse(str.substring(prefix.length, str.length - 1)),
        });
      }
    }

    return result;
  }

}
