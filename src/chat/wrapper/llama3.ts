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
import { ChatHistoryItem, ChatSystemMessage, ChatWrapper } from './../types';
import { LlamaContext } from '../../context/llama';
import { SpecialToken } from '../../types';

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

  generateContextState(ctx: LlamaContext, chatHistory: ChatHistoryItem[]) {

    const sys = _.first(chatHistory)?.type === 'system' ? _.first(chatHistory) as ChatSystemMessage : undefined;
    const history = sys ? _.drop(chatHistory, 1) : chatHistory;

    const sysMsg = sys?.text ?? (ctx.chatOptions?.functions ? [
      "The assistant calls the provided functions as needed to retrieve information instead of relying on existing knowledge.",
      "To fulfill a request, the assistant calls relevant functions in advance when needed before responding to the request, and does not tell the user prior to calling a function.",
      "Provided functions:",
      "```typescript",

      "```",
      "",
      "Calling any of the provided functions can be done like this:",
      "||call: getSomeInfo({ someKey: \"someValue\" })",
      "",
      "Note that the ||call: prefix is mandatory",
      "The assistant does not inform the user about using functions and does not explain anything before calling a function.",
      "After calling a function, the raw result appears afterwards and is not part of the conversation",
      "To make information be part of the conversation, the assistant paraphrases and repeats the information without the function syntax.",
    ] : [
      'You are a helpful AI assistant for travel tips and recommendations',
    ]).join('\n');

    const role = (role: string) => [SpecialToken('<|start_header_id|>'), role, SpecialToken('<|end_header_id|>'), '\n\n'];

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
  generateChatHistory(ctx: LlamaContext, tokens: Uint32Array): ChatHistoryItem[] {

    const beginOfText = ctx.model.tokenize(SpecialToken('<|begin_of_text|>'));
    const startHeaderId = ctx.model.tokenize(SpecialToken('<|start_header_id|>'));
    const endHeaderId = ctx.model.tokenize(SpecialToken('<|end_header_id|>'));

    if (!beginOfText.every((v, i) => tokens[i] === v)) throw Error('Invalid chat history');
    tokens = tokens.subarray(beginOfText.length);

    const result: ChatHistoryItem[] = [];

    while (tokens.length > 0) {

      if (!startHeaderId.every((v, i) => tokens[i] === v)) throw Error('Invalid chat history');
      tokens = tokens.subarray(startHeaderId.length);


    }

    return result;
  }

}
