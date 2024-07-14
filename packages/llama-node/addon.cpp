//
//  addon.cpp
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

#include "src/log.h"
#include "src/info.h"
#include "src/model.h"
#include "src/context.h"
#include "src/embedding.h"

Napi::Object registerCallback(Napi::Env env, Napi::Object exports)
{
  llama_backend_init();
  exports.DefineProperties({
      Napi::PropertyDescriptor::Function("systemInfo", systemInfo),
      Napi::PropertyDescriptor::Function("getSupportsGpuOffloading", getSupportsGpuOffloading),
      Napi::PropertyDescriptor::Function("getSupportsMmap", getSupportsMmap),
      Napi::PropertyDescriptor::Function("getSupportsMlock", getSupportsMlock),
      Napi::PropertyDescriptor::Function("getBlockSizeForGgmlType", getBlockSizeForGgmlType),
      Napi::PropertyDescriptor::Function("getTypeSizeForGgmlType", getTypeSizeForGgmlType),
      Napi::PropertyDescriptor::Function("getConsts", getConsts),
      Napi::PropertyDescriptor::Function("getGpuVramInfo", getGpuVramInfo),
      Napi::PropertyDescriptor::Function("getGpuDeviceInfo", getGpuDeviceInfo),
      Napi::PropertyDescriptor::Function("getGpuType", getGpuType),
  });
  LlamaModel::init(exports);
  LlamaContext::init(exports);
  LlamaEmbeddingContext::init(exports);
  llama_log_set(llama_log_callback, nullptr);
  return exports;
}

NODE_API_MODULE(NODE_GYP_MODULE_NAME, registerCallback)
