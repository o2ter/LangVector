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
import { LLMDevice } from './base';
import { LlamaModel } from '../model/llama';
import * as llamaCpp from '../plugins/llamaCpp';

export class LlamaDevice extends LLMDevice {

  static systemInfo() { return llamaCpp.systemInfo(); }
  static getSupportsGpuOffloading() { return llamaCpp.getSupportsGpuOffloading(); }
  static getSupportsMmap() { return llamaCpp.getSupportsMmap(); }
  static getSupportsMlock() { return llamaCpp.getSupportsMlock(); }
  static getBlockSizeForGgmlType(type: number) { return llamaCpp.getBlockSizeForGgmlType(type); }
  static getTypeSizeForGgmlType(type: number) { return llamaCpp.getTypeSizeForGgmlType(type); }
  static getConsts() { return llamaCpp.getConsts(); }
  static getGpuVramInfo() { return llamaCpp.getGpuVramInfo(); }
  static getGpuDeviceInfo() { return llamaCpp.getGpuDeviceInfo(); }
  static getGpuType() { return llamaCpp.getGpuType(); }

  static async loadModel({ modelPath, ...options }: {
    modelPath: string;
    gpuLayers?: number;
    vocabOnly?: boolean;
    useMmap?: boolean;
    useMlock?: boolean;
    checkTensors?: boolean;
  }) {
    const model = new llamaCpp.LlamaModel(modelPath, options);
    return new LlamaModel(this, model);
  }
}
