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

const defaultSysMsg = 'You are a helpful AI assistant for travel tips and recommendations';

export class Llama3ChatWrapper implements ChatWrapper {

  generateContextState(ctx: LlamaContext, chatHistory: ChatHistoryItem[]): { item: ChatHistoryItem; tokens: Uint32List; }[] {

    const shouldPrependBosToken = ctx.model.shouldPrependBosToken;
    const { bos, eot } = ctx.model.tokens;
    const start_header = ctx.model.tokenize('<|start_header_id|>', { encodeSpecial: true });
    const end_header = ctx.model.tokenize('<|start_header_id|>', { encodeSpecial: true });

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
      "Note that the || prefix is mandatory",
      "The assistant does not inform the user about using functions and does not explain anything before calling a function.",
      "After calling a function, the raw result appears afterwards and is not part of the conversation",
      "To make information be part of the conversation, the assistant paraphrases and repeats the information without the function syntax.",
    ].join('\n') : defaultSysMsg);

    const result: {
      item: ChatHistoryItem;
      tokens: Uint32List;
    }[] = [{
      item: {
        type: 'system',
        text: sysMsg,
      },
      tokens: ctx.model.tokenize([
        _.compact([shouldPrependBosToken && bos]),
        start_header, 'system', end_header, '\n\n',
        sysMsg,
        _.compact([eot]),
      ]),
    }];

    for (const item of history) {
      switch (item.type) {
        case 'user':
          result.push({
            item,
            tokens: ctx.model.tokenize([
              start_header, 'user', end_header, '\n\n',
              item.text,
              _.compact([eot]),
            ]),
          });
          break;
        case 'model':
          result.push({
            item,
            tokens: ctx.model.tokenize(_.flatMap(item.response, response => {
              if (_.isString(response)) {
                return [
                  start_header, 'assistant', end_header, '\n\n',
                  response,
                  _.compact([eot]),
                ];
              }
              return [
                start_header, 'assistant', end_header, '\n\n',
                '||call: ', response.name, '(', JSON.stringify(response.params), ')',
                start_header, 'function_call_result', end_header, '\n\n',
                JSON.stringify(response.result),
                _.compact([eot]),
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
    throw new Error('Method not implemented.');
  }

}
