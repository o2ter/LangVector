//
//  index.ts
//
//  The MIT License
//  Copyright (c) 2021 - 2025 O2ter Limited. All rights reserved.
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
import { Server } from '@o2ter/server-js';
import ProtoRoute from 'proto.io';
import { Proto } from './proto';
import './cloud/main';

import { createContext, models } from './llm';
import { LlamaContext } from '../../src';

export const serverOptions: Server.Options = {
  http: 'v1',
  express: {
    cors: {
      credentials: true,
      origin: true,
    },
    rateLimit: {
      windowMs: 1000,
      limit: 1000,
    },
  },
};

/* eslint-disable no-param-reassign */
export default async (app: Server, env: Record<string, any>) => {

  env.PROTO_ENDPOINT = 'http://localhost:8080/proto';

  app.express().use('/proto', await ProtoRoute({
    proto: Proto,
  }));

  app.socket().on('connection', async (socket) => {

    console.info('socket connected:', socket.id);

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

    socket.emit('response', {
      status: 'initiating',
      models,
      options,
    });

    const abort = new AbortController;

    let currentModel = 'meta-llama/Meta-Llama-3.1-8B-Instruct/ggml-model-q3_k_m.gguf';
    let session = currentModel ? await createContext(currentModel) : null;

    const defaultResponse = (session: LlamaContext) => ({
      models,
      currentModel,
      options,
      history: session.chatHistory,
      tokens: session.tokens.length,
      contextSize: session.contextSize,
      maxContextSize: session.maxContextSize,
      raw: session.model.detokenize(session.tokens, { decodeSpecial: true }),
    });

    if (session) {
      socket.emit('response', {
        status: 'ready',
        ...defaultResponse(session),
      });
    }

    socket.on('reset', async () => {
      session?.dispose();
      session = await createContext(currentModel);
    });

    socket.on('sync', (opts: any) => {
      if (opts?.minP) options.minP = opts.minP;
      if (opts?.topK) options.topK = opts.topK;
      if (opts?.topP) options.topP = opts.topP;
      if (opts?.temperature) options.temperature = opts.temperature;
      if (opts?.repeatPenalty?.frequencyPenalty) options.repeatPenalty.frequencyPenalty = opts.repeatPenalty?.frequencyPenalty;
      if (opts?.repeatPenalty?.presencePenalty) options.repeatPenalty.presencePenalty = opts.repeatPenalty?.presencePenalty;
      if (opts?.maxTokens) options.maxTokens = opts.maxTokens;

      const _session = session;
      if (!_session) return;

      socket.emit('response', {
        status: 'ready',
        ...defaultResponse(_session),
      });
    });

    socket.on('msg', async (msg: string) => {

      const _session = session;
      if (!_session) return;

      socket.emit('response', {
        status: 'responding',
        ...defaultResponse(_session),
        partial: true,
        message: msg,
        responseText: '',
      });

      const generator = _session.prompt(msg, {
        ...options,
        signal: abort.signal,
      });

      for await (const { response, done } of generator) {
        socket.emit('response', {
          status: done ? 'ready' : 'responding',
          ...defaultResponse(_session),
          partial: !done,
          message: msg,
          responseText: _session.model.detokenize(response, { decodeSpecial: true }),
        });
      }
    });

    socket.on('disconnect', () => {
      console.info('socket disconnected:', socket.id);
      abort.abort();
      if (!session) return;
      session.dispose();
      session = null;
    });
  });

}
