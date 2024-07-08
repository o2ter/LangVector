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

import { defineChatSessionFunction, LLMDevice, Token } from '../../src';
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
        console.log('function called');
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
        console.log('function called');
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
  try {
    for await (const file of walkDirAsync(path.join(__dirname, '../../models'))) {
      if (file.endsWith('.gguf')) {
        models.push(file);
      }
    }
  } catch { }

  const device = await LLMDevice.llama();
  const contexts: Record<string, LlamaContext> = {};

  const createSession = async (modelPath: string) => {
    if (contexts[modelPath]) return contexts[modelPath].createSession();
    const model = await device.loadModel({ modelPath });
    const context = await model.createContext();
    contexts[modelPath] = context;
    return context.createSession();
  }

  app.socket().on('connection', async (socket) => {

    let session = models.length ? await createSession(models[0]) : null;

    const options = {
      ...defaultOptions,
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
      if (opts.frequencyPenalty) options.repeatPenalty.frequencyPenalty = opts.frequencyPenalty;
      if (opts.presencePenalty) options.repeatPenalty.presencePenalty = opts.presencePenalty;
      if (opts.maxTokens) options.maxTokens = opts.maxTokens;
    });

    socket.on('sync', () => {

      const _session = session;
      if (!_session) return;

      socket.emit('response', {
        history: _session.chatHistory,
        raw: _session.model.detokenize(_session.tokens, true),
      });
    });

    socket.on('msg', async (msg: string) => {

      const _session = session;
      if (!_session) return;

      let partial: Token[] = [];

      const { responseText } = await _session.prompt(msg, {
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
        partial: false,
        responseText,
        history: _session.chatHistory,
        raw: _session.model.detokenize(_session.tokens, true),
      });
    });

    socket.on('disconnect', () => {
      if (!session) return;
      session.dispose();
      session = null;
    });
  });

}
