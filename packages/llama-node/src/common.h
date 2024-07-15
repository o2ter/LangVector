//
//  common.h
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

#pragma once

#include <stddef.h>

#include "llama.h"
#include "common/common.h"
#include "napi.h"

#ifdef GPU_INFO_USE_CUDA
#include "../gpuInfo/cuda-gpu-info.h"
#endif
#ifdef GPU_INFO_USE_VULKAN
#include "../gpuInfo/vulkan-gpu-info.h"
#endif
#ifdef GPU_INFO_USE_METAL
#include "../gpuInfo/metal-gpu-info.h"
#endif

class _AsyncWorker : public Napi::AsyncWorker
{
public:
  _AsyncWorker(
      Napi::Env env,
      std::function<void()> execute,
      std::function<void()> finalizer = []() {}) : AsyncWorker(env), execute(execute), finalizer(finalizer), deferred(Napi::Promise::Deferred::New(env))
  {
  }
  ~_AsyncWorker()
  {
    finalizer();
  }

  Napi::Promise Promise()
  {
    return deferred.Promise();
  }

private:
  std::function<void()> execute;
  std::function<void()> finalizer;
  Napi::Promise::Deferred deferred;

  void Execute()
  {
    try
    {
      execute();
    }
    catch (const std::exception &e)
    {
      SetError(e.what());
    }
    catch (...)
    {
      SetError("Unknown error");
    }
  }

  void OnOK()
  {
    deferred.Resolve(Env().Undefined());
  }

  void OnError(const Napi::Error &err)
  {
    deferred.Reject(err.Value());
  }
};

template <typename R>
class _AsyncWorkerWithResult : public Napi::AsyncWorker
{
public:
  _AsyncWorker(
      Napi::Env env,
      std::function<R()> execute,
      std::function<napi_value(R result)> resolve,
      std::function<void()> finalizer = []() {}) : AsyncWorker(env), execute(execute), finalizer(finalizer), deferred(Napi::Promise::Deferred::New(env))
  {
  }
  ~_AsyncWorker()
  {
    finalizer();
  }

  Napi::Promise Promise()
  {
    return deferred.Promise();
  }

private:
  std::function<R()> execute;
  std::function<napi_value(R result)> resolve;
  std::function<void()> finalizer;
  Napi::Promise::Deferred deferred;
  R result;

  void Execute()
  {
    try
    {
      result = execute();
    }
    catch (const std::exception &e)
    {
      SetError(e.what());
    }
    catch (...)
    {
      SetError("Unknown error");
    }
  }

  void OnOK()
  {
    deferred.Resolve(resolve(result));
  }

  void OnError(const Napi::Error &err)
  {
    deferred.Reject(err.Value());
  }
};
