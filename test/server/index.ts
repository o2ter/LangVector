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

import { LLMDevice } from '../../src';
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
    const model = await device.loadModel({ modelPath: models[0] });
    const context = await model.createContext();
    contexts[modelPath] = context;
    return context.createSession();
  }

  app.socket().on('connection', async (socket) => {

    let session = models.length ? await createSession(models[0]) : null;

    socket.on('msg', async (msg: string) => {

      if (session) {

      }

    })

    socket.on('disconnect', () => {

      if (session) {
        session.dispose();
        session = null;
      }
    });
  });

}
