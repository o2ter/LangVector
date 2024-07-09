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

import { LlamaModel } from '../model/llama';
import { LLMDeviceBase } from './base';
import { llamaCpp } from '../plugins/llama-cpp';

export class LlamaDevice extends LLMDeviceBase<llamaCpp.Llama> {

  async dispose() {
    if (!this._device.disposed) await this._device.dispose();
  }

  get disposed() {
    return this._device.disposed;
  }

  async loadModel(options: llamaCpp.LlamaModelOptions) {
    const model = await this._device.loadModel(options);
    return new LlamaModel(this, model);
  }

  get gpu() {
    return this._device.gpu;
  }
  get supportsGpuOffloading() {
    return this._device.supportsGpuOffloading;
  }
  get supportsMmap() {
    return this._device.supportsMmap;
  }
  get supportsMlock() {
    return this._device.supportsMlock;
  }
  get logLevel() {
    return this._device.logLevel;
  }
  set logLevel(value: llamaCpp.LlamaLogLevel) {
     this._device.logLevel = value;
  }
  get logger() {
    return this._device.logger;
  }
  set logger(value: (level: llamaCpp.LlamaLogLevel, message: string) => void) {
    this._device.logger = value;
  }
  get buildType() {
    return this._device.buildType;
  }
  get cmakeOptions() {
    return this._device.cmakeOptions;
  }
  get llamaCppRelease() {
    return this._device.llamaCppRelease;
  }
  get systemInfo() {
    return this._device.systemInfo;
  }
  
  get vramPaddingSize() {
    return this._device.vramPaddingSize;
  }
  getVramState() {
    return this._device.getVramState;
  }
  getGpuDeviceNames() {
    return this._device.getGpuDeviceNames;
  }
}
