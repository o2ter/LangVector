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
import { LlamaDevice } from '../device/llama';
import { LlamaModel } from '../model/llama';
import { LLMSession } from './base';
import { llamaCpp } from '../plugins/llama-cpp';

export class LlamaSession extends LLMSession<LlamaDevice, LlamaModel, LlamaContext> {

  private _seq: llamaCpp.LlamaContextSequence;
  private _chat?: llamaCpp.LlamaChatSession;
  private _options: Omit<llamaCpp.LlamaChatSessionOptions, 'contextSequence'>;

  constructor(
    context: LlamaContext,
    seq: llamaCpp.LlamaContextSequence,
    options: Omit<llamaCpp.LlamaChatSessionOptions, 'contextSequence'>,
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

  get nextTokenIndex() {
    return this._seq.nextTokenIndex;
  }
  get tokens() {
    return this._seq.contextTokens;
  }
  get tokenMeter() {
    return this._seq.tokenMeter;
  }
  get isLoadedToMemory() {
    return this._seq.isLoadedToMemory;
  }

  compareContextTokens(tokens: llamaCpp.Token[]) {
    return this._seq.compareContextTokens(tokens);
  }

  async clearHistory() {
    if (this._chat) this._chat.setChatHistory([]);
    await this._seq.clearHistory();
  }

  async eraseContextTokenRanges(ranges: llamaCpp.ContextTokensDeleteRange[]) {
    await this._seq.eraseContextTokenRanges(ranges);
  }

  evaluate(
    tokens: llamaCpp.Token[],
    options?: Parameters<llamaCpp.LlamaContextSequence['evaluate']>[1]
  ) {
    return this._seq.evaluate(tokens, options);
  }

  evaluateWithoutGeneratingNewTokens(
    tokens: llamaCpp.Token[],
    options?: Parameters<llamaCpp.LlamaContextSequence['evaluateWithoutGeneratingNewTokens']>[1]
  ) {
    return this._seq.evaluateWithoutGeneratingNewTokens(tokens, options);
  }

  get #chat() {
    this._chat = this._chat ?? new llamaCpp.LlamaChatSession({ contextSequence: this._seq, ...this._options });
    return this._chat;
  }

  prompt<const Functions extends llamaCpp.ChatSessionModelFunctions | undefined = undefined>(
    prompt: string,
    options?: llamaCpp.LLamaChatPromptOptions<Functions>
  ) {
    return this.#chat.promptWithMeta(prompt, options);
  }

  completePrompt(
    prompt: string,
    options?: llamaCpp.LLamaChatCompletePromptOptions
  ) {
    return this.#chat.completePromptWithMeta(prompt, options);
  }

  get chatHistory() {
    return this.#chat.getChatHistory();
  }
  set chatHistory(history: llamaCpp.ChatHistoryItem[]) {
    this.#chat.setChatHistory(history);
  }
  get lastEvaluationContextWindow() {
    return this.#chat.getLastEvaluationContextWindow();
  }
}
