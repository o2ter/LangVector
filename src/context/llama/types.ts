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

import { Awaitable } from "@o2ter/utils-js";
import { ChatWrapper } from "../../chat/types";
import { LLMTextValue } from "../../types";
import type { LlamaContext } from "./index";

type TokenBias = {
  tokens: LLMTextValue;
  bias: 'never' | number;
}[];

export type LlamaContextOptions = {
  seed?: number;
  /**
   * The context size of context. (default to model's context size)
   */
  contextSize?: number;
  /**
   * Max number of tokens in a single batch.
   */
  batchSize?: number;
  flashAttention?: boolean;
  /**
   * Max number of threads. (default to hardware)
   */
  threads?: number;

  chatOptions?: {
    contextCompression?: (session: LlamaContext) => Awaitable<Uint32List>;
    chatWrapper?: ChatWrapper;
  };
};

export type LlamaSequenceRepeatPenalty = {
  /** Tokens to lower the predication probability of to be the next predicted token */
  punishTokens: Uint32List | (() => Uint32List);
  /**
   * The relative amount to lower the probability of the tokens in `punishTokens` by
   * Defaults to `1.1`.
   * Set to `1` to disable.
   */
  penalty?: number;
  /**
   * For n time a token is in the `punishTokens` array, lower its probability by `n * frequencyPenalty`
   * Disabled by default (`0`).
   * Set to a value between `0` and `1` to enable.
   */
  frequencyPenalty?: number;
  /**
   * Lower the probability of all the tokens in the `punishTokens` array by `presencePenalty`
   * Disabled by default (`0`).
   * Set to a value between `0` and `1` to enable.
   */
  presencePenalty?: number;
};

export type LLamaChatPromptOptions = {

  onToken?: (tokens: Uint32Array) => void;

  signal?: AbortSignal;

  maxTokens?: number;
  /**
   * Temperature is a hyperparameter that controls the randomness of the generated text.
   * It affects the probability distribution of the model's output tokens.
   * A higher temperature (e.g., 1.5) makes the output more random and creative,
   * while a lower temperature (e.g., 0.5) makes the output more focused, deterministic, and conservative.
   * The suggested temperature is 0.8, which provides a balance between randomness and determinism.
   * At the extreme, a temperature of 0 will always pick the most likely next token, leading to identical outputs in each run.
   *
   * Set to `0` to disable.
   * Disabled by default (set to `0`).
   */
  temperature?: number;
  /**
   * From the next token candidates, discard the percentage of tokens with the lowest probability.
   * For example, if set to `0.05`, 5% of the lowest probability tokens will be discarded.
   * This is useful for generating more high-quality results when using a high temperature.
   * Set to a value between `0` and `1` to enable.
   *
   * Only relevant when `temperature` is set to a value greater than `0`.
   * Disabled by default.
   */
  minP?: number;
  /**
   * Limits the model to consider only the K most likely next tokens for sampling at each step of sequence generation.
   * An integer number between `1` and the size of the vocabulary.
   * Set to `0` to disable (which uses the full vocabulary).
   *
   * Only relevant when `temperature` is set to a value greater than 0.
   */
  topK?: number;
  /**
   * Dynamically selects the smallest set of tokens whose cumulative probability exceeds the threshold P,
   * and samples the next token only from this set.
   * A float number between `0` and `1`.
   * Set to `1` to disable.
   *
   * Only relevant when `temperature` is set to a value greater than `0`.
   */
  topP?: number;
  /**
   * Trim whitespace from the end of the generated text
   * Disabled by default.
   */
  trimWhitespaceSuffix?: boolean;

  repeatPenalty?: false | LlamaSequenceRepeatPenalty;
  /**
   * Adjust the probability of tokens being generated.
   * Can be used to bias the model to generate tokens that you want it to lean towards,
   * or to avoid generating tokens that you want it to avoid.
   */
  tokenBias?: () => TokenBias;
  /**
   * Custom stop triggers to stop the generation of the response when any of the provided triggers are found.
   */
  stopTriggers?: LLMTextValue[];
};