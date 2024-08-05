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

import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

let pkg: any;

try {
  const require = createRequire(import.meta.url);
  pkg = require('../../packages/llama-node/build/Release/llama-node.node');
} catch {}

if (!pkg) {

  const modulePath = (() => {
    const dir = import.meta.url.replace('file://', '').split('/');
    while (dir.length) {
      const node_path = path.resolve('/', ...dir, 'packages/llama-node/build/Release/llama-node.node');
      if (fs.existsSync(node_path)) return node_path;
      dir.pop();
    }
    throw Error('llama-node not found.');
  })();

  if (typeof require === 'undefined') {
    const require = createRequire(import.meta.url);
    pkg = require(modulePath);
  } else {
    pkg = eval(`require('${modulePath}')`);
  }
}

export const LlamaModel = pkg.LlamaModel;
export const LlamaContext = pkg.LlamaContext;
export const LlamaContextSampleCandidates = pkg.LlamaContextSampleCandidates;
export const LlamaEmbeddingContext = pkg.LlamaEmbeddingContext;
export const LlamaGrammar = pkg.LlamaGrammar;
export const LlamaGrammarEvaluationState = pkg.LlamaGrammarEvaluationState;

export const systemInfo = (): string => {
  return pkg.systemInfo();
};

export const getGpuVramInfo = (): { total: number; used: number; } => {
  return pkg.getGpuVramInfo();
};

export const getGpuDeviceInfo = (): { deviceNames: string[]; } => {
  return pkg.getGpuDeviceInfo();
};

export const getGpuType = (): string | undefined => {
  return pkg.getGpuType();
};

export const getSupportsGpuOffloading = (): boolean => {
  return pkg.getSupportsGpuOffloading();
};

export const getSupportsMmap = (): boolean => {
  return pkg.getSupportsMmap();
};

export const getSupportsMlock = (): boolean => {
  return pkg.getSupportsMlock();
};

export const getBlockSizeForGgmlType = (type: number): number => {
  return pkg.getBlockSizeForGgmlType(type);
};

export const getTypeSizeForGgmlType = (type: number): number => {
  return pkg.getTypeSizeForGgmlType(type);
};

export const getConsts = (): Record<string, number> => {
  return pkg.getConsts();
};