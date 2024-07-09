//
//  index.ts
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
import path from 'path';
import fs from 'fs/promises';
import { Server } from '@o2ter/server-js';
import ProtoRoute from 'proto.io';
import { Proto } from './proto';
import './cloud/main';

import { defineChatSessionFunction, LLMDevice, Token, llamaCpp } from '../../src';
import { LlamaContext } from '../../src/llm/context/llama';

const walkDirAsync = async function* (dir: string): AsyncGenerator<string, void> {
  const files = await fs.readdir(dir, { withFileTypes: true });
  for (const file of files) {
    if (file.isDirectory()) {
      yield* walkDirAsync(path.join(dir, file.name));
    } else {
      yield path.join(dir, file.name);
    }
  }
}

const defaultOptions = {
  documentFunctionParams: true,
  functions: {
    datetime: defineChatSessionFunction({
      description: "Get current datetime",
      handler() {
        return new Date();
      }
    }),
    random: defineChatSessionFunction({
      description: "Generates a random number",
      params: {
        type: 'object',
        properties: {
          maximum: { type: 'number' },
          minimum: { type: 'number' },
        },
        required: ['maximum', 'minimum'],
      },
      handler({ maximum, minimum }) {
        return Math.random() * (maximum - minimum) + minimum;
      }
    })
  }
};

/* eslint-disable no-param-reassign */
export default async (app: Server, env: Record<string, any>) => {

  env.PROTO_ENDPOINT = 'http://localhost:8080/proto';

  app.express().use('/proto', await ProtoRoute({
    proto: Proto,
  }));

  let models = [];
  const modelsDir = path.join(__dirname, '../../models');
  try {
    for await (const file of walkDirAsync(modelsDir)) {
      if (file.endsWith('.gguf')) {
        models.push(file.slice(modelsDir.length + 1));
      }
    }
  } catch { }

  const device = await LLMDevice.llama();
  const contexts: Record<string, LlamaContext> = {};
  const chatOptions = { chatWrapper: new llamaCpp.Llama3ChatWrapper };

  const createSession = async (name: string) => {
    if (contexts[name]) return contexts[name].createSession({ chatOptions });
    const model = await device.loadModel({
      modelPath: path.join(modelsDir, name),
      ignoreMemorySafetyChecks: true,
     });
    const context = await model.createContext({ ignoreMemorySafetyChecks: true });
    contexts[name] = context;
    return context.createSession({ chatOptions });
  }

  app.socket().on('connection', async (socket) => {

    console.log('socket connected:', socket.id);

    let currentModel = 'meta-llama/Meta-Llama-3-8B-Instruct/ggml-model-q3_k_m.gguf';
    let session = currentModel ? await createSession(currentModel) : null;

    const options = {
      minP: 0,
      topK: 40,
      topP: 0.75,
      temperature: 0.8,
      repeatPenalty: {
        frequencyPenalty: 0.2,
        presencePenalty: 0.2,
      },
      maxTokens: 100,
    };

    socket.on('options', (opts: any) => {
      if (opts.minP) options.minP = opts.minP;
      if (opts.topK) options.topK = opts.topK;
      if (opts.topP) options.topP = opts.topP;
      if (opts.temperature) options.temperature = opts.temperature;
      if (opts.repeatPenalty?.frequencyPenalty) options.repeatPenalty.frequencyPenalty = opts.repeatPenalty?.frequencyPenalty;
      if (opts.repeatPenalty?.presencePenalty) options.repeatPenalty.presencePenalty = opts.repeatPenalty?.presencePenalty;
      if (opts.maxTokens) options.maxTokens = opts.maxTokens;
    });

    socket.on('reset', () => {
      session?.clearHistory();
    });

    socket.on('sync', () => {

      const _session = session;
      if (!_session) return;

      socket.emit('response', {
        models,
        currentModel,
        options,
        history: _session.chatHistory,
        raw: _session.model.detokenize(_session.tokens, true),
      });
    });

    socket.on('msg', async (msg: string) => {

      const _session = session;
      if (!_session) return;

      let partial: Token[] = [];

      const { responseText } = await _session.prompt(msg, {
        ...defaultOptions,
        ...options,
        onToken: (token) => {
          partial.push(...token);
          socket.emit('response', {
            partial: true,
            responseText: _session.model.detokenize(partial, true),
          });
        }
      });

      socket.emit('response', {
        models,
        currentModel,
        options,
        history: _session.chatHistory,
        raw: _session.model.detokenize(_session.tokens, true),
        partial: false,
        responseText,
      });
    });

    socket.on('disconnect', () => {
      if (!session) return;
      session.dispose();
      session = null;
    });
  });

}
