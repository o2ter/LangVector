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

import { LLMTextValue } from '../types';
import { LlamaGrammar } from '../device/llama/grammar';
import type { LlamaContext } from '../context/llama';

export type ChatHistoryItem = ChatSystemMessage | ChatUserMessage | ChatModelResponse;
export type ChatSystemMessage = {
  type: "system";
  text: LLMTextValue;
};
export type ChatUserMessage = {
  type: "user";
  text: string;
};
export type ChatModelResponse = {
  type: "model";
  response: (string | ChatModelFunctionCall)[];
};
export type ChatModelFunctionCall = {
  type: "functionCall";
  name: string;
  description?: string;
  params: any;
  result: any;
};

export type ChatWrapper = {

  stopGenerationTriggers: (ctx: LlamaContext) => LLMTextValue[];

  generateFunctionGrammar?: (
    ctx: LlamaContext,
  ) => {
    beginTrigger: Uint32List;
    grammar: () => LlamaGrammar;
    stopGenerationTriggers: Uint32List[];
    responseEncoder: (result: any) => string;
  } | undefined;

  encodeNextContextState(ctx: LlamaContext, role: string, value: LLMTextValue): LLMTextValue;

  encodeContextState: (
    ctx: LlamaContext,
    chatHistory: ChatHistoryItem[],
  ) => { item: ChatHistoryItem; tokens: Uint32List; }[];

  decodeChatHistory: (
    ctx: LlamaContext,
    tokens: Uint32Array,
  ) => ChatHistoryItem[];

  decodeFunctionCalls(
    ctx: LlamaContext,
    tokens: string,
  ): { name: string; params: any; }[];

};
