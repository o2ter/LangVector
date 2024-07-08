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

import { LlamaContext } from '../context/llama';
import {
  LlamaModel as _LlamaModel,
  ChatSessionModelFunctions,
  ContextTokensDeleteRange,
  LLamaChatCompletePromptOptions,
  LLamaChatPromptOptions,
  LlamaChatSession,
  LlamaChatSessionOptions,
  LlamaContextSequence,
  Token,
} from '../plugins/llama-cpp';
import { LLMSession } from './index';

export class LlamaSession extends LLMSession<LlamaContext> {

  private _seq: LlamaContextSequence;
  private _chat?: LlamaChatSession;
  private _options: Omit<LlamaChatSessionOptions, 'contextSequence'>;

  constructor(
    context: LlamaContext,
    seq: LlamaContextSequence,
    options: Omit<LlamaChatSessionOptions, 'contextSequence'>,
  ) {
    super(context);
    this._seq = seq;
    this._options = options;
  }

  async dispose() {
    if (!this._seq.disposed) await this._seq.dispose();
    if (this._chat && !this._chat.disposed) await this._chat.dispose();
  }

  get disposed() {
    if (this._chat && !this._chat.disposed) return false;
    return this._seq.disposed;
  }

  async clearHistory() {
    if (this._chat) this._chat.setChatHistory([]);
    await this._seq.clearHistory();
  }

  async eraseContextTokenRanges(ranges: ContextTokensDeleteRange[]) {
    await this._seq.eraseContextTokenRanges(ranges);
  }

  evaluate(
    tokens: Token[],
    options?: Parameters<LlamaContextSequence['evaluate']>[1]
  ) {
    return this._seq.evaluate(tokens, options);
  }

  evaluateWithoutGeneratingNewTokens(
    tokens: Token[],
    options?: Parameters<LlamaContextSequence['evaluateWithoutGeneratingNewTokens']>[1]
  ) {
    return this._seq.evaluateWithoutGeneratingNewTokens(tokens, options);
  }

  prompt<const Functions extends ChatSessionModelFunctions | undefined = undefined>(
    prompt: string,
    options?: LLamaChatPromptOptions<Functions>
  ) {
    this._chat = this._chat ?? new LlamaChatSession({ contextSequence: this._seq, ...this._options });
    return this._chat.promptWithMeta(prompt, options);
  }

  completePrompt(
    prompt: string,
    options?: LLamaChatCompletePromptOptions
  ) {
    this._chat = this._chat ?? new LlamaChatSession({ contextSequence: this._seq, ...this._options });
    return this._chat.completePromptWithMeta(prompt, options);
  }
}
