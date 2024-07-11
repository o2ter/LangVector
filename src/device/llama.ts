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

import _ from 'lodash';
import path from 'path';
import { LLMDevice } from './base';
import { LlamaModel } from '../model/llama';
import * as llamaCpp from '../plugins/llamaCpp';

export class LlamaDevice extends LLMDevice {

  static systemInfo() { return llamaCpp.systemInfo(); }
  static supportsGpuOffloading() { return llamaCpp.getSupportsGpuOffloading(); }
  static supportsMmap() { return llamaCpp.getSupportsMmap(); }
  static supportsMlock() { return llamaCpp.getSupportsMlock(); }
  static blockSizeForGgmlType(type: number) { return llamaCpp.getBlockSizeForGgmlType(type); }
  static typeSizeForGgmlType(type: number) { return llamaCpp.getTypeSizeForGgmlType(type); }
  static consts() { return llamaCpp.getConsts(); }
  static gpuVramInfo() { return llamaCpp.getGpuVramInfo(); }
  static gpuDeviceInfo() { return llamaCpp.getGpuDeviceInfo(); }
  static gpuType() { return llamaCpp.getGpuType(); }

  static loadModel({ modelPath, onLoadProgress, ...options }: {
    modelPath: string;
    gpuLayers?: number;
    vocabOnly?: boolean;
    useMmap?: boolean;
    useMlock?: boolean;
    checkTensors?: boolean;
    onLoadProgress?: (progress: number) => boolean;
  }) {
    const model = new llamaCpp.LlamaModel(
      path.resolve(process.cwd(), modelPath),
      {
        ...options,
        onLoadProgress: onLoadProgress ? (progress: number) => {
          return !!onLoadProgress(progress);
        } : null,
      }
    );
    return new LlamaModel(this, model);
  }
}
