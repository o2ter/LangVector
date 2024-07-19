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

const role = (role: string) => [SpecialToken('<|start_header_id|>'), role, SpecialToken('<|end_header_id|>'), '\n\n'];

const find = (tokens: Uint32Array, pattern: Uint32Array) => {
  for (let offset = 0; offset < tokens.length; ++offset) {
    if (offset + pattern.length > tokens.length) return -1;
    if (pattern.every((v, i) => tokens[i + offset] === v)) return offset;
  }
  return -1;
};

const startsWith = (tokens: Uint32Array, pattern: Uint32Array) => {
  if (tokens.length < pattern.length) return false;
  return pattern.every((v, i) => tokens[i] === v);
};

const endsWith = (tokens: Uint32Array, pattern: Uint32Array) => {
  if (tokens.length < pattern.length) return false;
  const offset = tokens.length - pattern.length;
  return pattern.every((v, i) => tokens[i + offset] === v);
};

export class Llama3ChatWrapper implements ChatWrapper {

  stopGenerationTriggers(ctx: LlamaContext) {
    const { eot, eos } = ctx.model.tokens;
    return _.filter([
      _.compact([eot]),
      _.compact([eos]),
      SpecialToken('<|eot_id|>'),
      SpecialToken('<|end_of_text|>'),
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

  encodeNextContextState(ctx: LlamaContext, value: LLMTextValue) {
    const eot = ctx.model.tokenize(SpecialToken('<|eot_id|>'));
    return _.compact([
      _.isEmpty(ctx._tokens) && [
        SpecialToken('<|begin_of_text|>'),
        role('system'),
        this._defaultSystemMessage(ctx),
        SpecialToken('<|eot_id|>'),
      ],
      !_.isEmpty(ctx._tokens) && !endsWith(ctx.tokens, eot) && eot,
      role('user'),
      value,
      SpecialToken('<|eot_id|>'),
      role('assistant'),
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
        SpecialToken('<|begin_of_text|>'),
        role('system'),
        sysMsg,
        SpecialToken('<|eot_id|>'),
      ]),
    }];

    for (const item of history) {
      switch (item.type) {
        case 'user':
          result.push({
            item,
            tokens: ctx.model.tokenize([
              role('user'),
              item.text,
              SpecialToken('<|eot_id|>'),
            ]),
          });
          break;
        case 'model':
          result.push({
            item,
            tokens: ctx.model.tokenize(_.flatMap(item.response, response => {
              if (_.isString(response)) {
                return [
                  role('assistant'),
                  response,
                  SpecialToken('<|eot_id|>'),
                ];
              }
              return [
                role('assistant'),
                `||call: ${response.name}(${JSON.stringify(response.params)})`,
                SpecialToken('<|eot_id|>'),
                role('function_call_result'),
                JSON.stringify(response.result),
                SpecialToken('<|eot_id|>'),
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
    const beginOfText = ctx.model.tokenize(SpecialToken('<|begin_of_text|>'));
    const startHeaderId = ctx.model.tokenize(SpecialToken('<|start_header_id|>'));
    const endHeaderId = ctx.model.tokenize(SpecialToken('<|end_header_id|>'));
    const eotId = ctx.model.tokenize(SpecialToken('<|eot_id|>'));

    if (!startsWith(tokens, beginOfText)) throw Error('Invalid chat history');
    tokens = tokens.subarray(beginOfText.length);

    const result: ChatHistoryItem[] = [];

    while (tokens.length > 0) {

      if (!startsWith(tokens, startHeaderId)) throw Error('Invalid chat history');
      tokens = tokens.subarray(startHeaderId.length);

      const endHeaderIdx = find(tokens, endHeaderId);
      if (endHeaderIdx === -1) throw Error('Invalid chat history');

      const type = tokens.subarray(0, endHeaderIdx);
      tokens = tokens.subarray(endHeaderIdx + endHeaderId.length);

      while (startsWith(tokens, newline)) {
        tokens = tokens.subarray(newline.length);
      }

      const _end = find(tokens, eotId);
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
            const last = _.last(result);
            if (last?.type !== 'model') throw Error('Invalid chat history');

            const calls = _.filter(last.response, x => _.isString(x) && _.startsWith(x, '||call: ')) as string[];
            const results = _.filter(last.response, x => !_.isString(x) && x.type === 'functionCall') as ChatModelFunctionCall[];

              // result.push({
              //   type: 'model',
              //   response: [{
              //     type: 'functionCall',
              //   }],
              // });
          }
          break;
        default: throw Error('Invalid chat history');
      }
    }

    return result;
  }

}
