//
//  context.h
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

#include "common.h"
#include "model.h"

class LlamaContext : public Napi::ObjectWrap<LlamaContext>
{
public:
  LlamaModel *model;
  llama_context_params params;
  llama_context *ctx;

  LlamaContext(const Napi::CallbackInfo &info) : Napi::ObjectWrap<LlamaContext>(info)
  {
    model = Napi::ObjectWrap<LlamaModel>::Unwrap(info[0].As<Napi::Object>());
    model->Ref();

    auto hardware_concurrency = std::thread::hardware_concurrency();

    params = llama_context_default_params();
    params.seed = -1;
    params.n_ctx = 0;
    params.n_seq_max = 1;
    params.n_threads = hardware_concurrency;
    params.n_threads_batch = hardware_concurrency;

    Napi::Object options = info[1].As<Napi::Object>();

    if (options.Has("seed") && options.Get("seed").IsNumber())
    {
      params.seed = options.Get("seed").As<Napi::Number>().Uint32Value();
    }

    if (options.Has("contextSize") && options.Get("contextSize").IsNumber())
    {
      params.n_ctx = options.Get("contextSize").As<Napi::Number>().Uint32Value();
    }

    if (options.Has("batchSize") && options.Get("batchSize").IsNumber())
    {
      params.n_batch = options.Get("batchSize").As<Napi::Number>().Uint32Value();
      params.n_ubatch = params.n_batch;
    }

    if (options.Has("flashAttention") && options.Get("flashAttention").IsBoolean())
    {
      params.flash_attn = options.Get("flashAttention").As<Napi::Boolean>().Value();
    }

    if (options.Has("threads") && options.Get("threads").IsNumber())
    {
      const auto n_threads = options.Get("threads").As<Napi::Number>().Uint32Value();
      const auto resolved_n_threads = n_threads > 0 ? n_threads : hardware_concurrency;
      params.n_threads = resolved_n_threads;
      params.n_threads_batch = resolved_n_threads;
    }

    ctx = llama_new_context_with_model(model->model, params);
    if (ctx == NULL)
    {
      Napi::Error::New(Env(), "Failed to load context").ThrowAsJavaScriptException();
    }

    Napi::MemoryManagement::AdjustExternalMemory(Env(), llama_state_get_size(ctx));
  }

  ~LlamaContext()
  {
    dispose();
  }

  void dispose()
  {
    if (ctx == NULL)
    {
      return;
    }
    Napi::MemoryManagement::AdjustExternalMemory(Env(), -(int64_t)llama_state_get_size(ctx));
    llama_free(ctx);
    ctx = NULL;
    model->Unref();
  }

  Napi::Value Dispose(const Napi::CallbackInfo &info)
  {
    if (ctx == NULL)
    {
      return Env().Undefined();
    }
    dispose();
  }

  Napi::Value GetContextSize(const Napi::CallbackInfo &info)
  {
    return Napi::Number::From(Env(), llama_n_ctx(ctx));
  }

  Napi::Value GetStateSize(const Napi::CallbackInfo &info)
  {
    return Napi::Number::From(Env(), llama_state_get_size(ctx));
  }

  Napi::Value EvalSequence(const Napi::CallbackInfo &info)
  {
    Napi::Uint32Array tokens = info[0].As<Napi::Uint32Array>();

    this->Ref();

    auto worker = new _AsyncWorker(
        Env(),
        [=]()
        {
          auto token_length = tokens.ElementLength();
          size_t n_batch = llama_n_batch(ctx);

          if (token_length > n_batch)
          {
            throw std::runtime_error("error: number of tokens exceeds batch size");
          }

          llama_batch batch = llama_batch_init(token_length, 0, 1);

          for (size_t i = 0; i < token_length; ++i)
          {
            llama_batch_add(batch, tokens[i], i, {0}, false);
          }
          auto status = llama_decode(ctx, batch);
          if (status < 0)
          {
            throw std::runtime_error("Eval failed");
          }

          llama_synchronize(ctx);
          llama_batch_free(batch);
        },
        [=]()
        {
          this->Unref();
        });

    worker->Queue();
    return worker->Promise();
  }

  Napi::Value RemoveTokens(const Napi::CallbackInfo &info)
  {
    int32_t startPos = info[0].As<Napi::Number>().Int32Value();
    int32_t endPos = info[1].As<Napi::Number>().Int32Value();

    bool result = llama_kv_cache_seq_rm(ctx, 0, startPos, endPos);

    return Napi::Boolean::New(info.Env(), result);
  }
  Napi::Value ShiftTokens(const Napi::CallbackInfo &info)
  {
    int32_t startPos = info[0].As<Napi::Number>().Int32Value();
    int32_t endPos = info[1].As<Napi::Number>().Int32Value();
    int32_t shiftDelta = info[2].As<Napi::Number>().Int32Value();

    llama_kv_cache_seq_add(ctx, 0, startPos, endPos, shiftDelta);

    return info.Env().Undefined();
  }
  static void init(Napi::Object exports)
  {
    auto def = DefineClass(
        exports.Env(),
        "LlamaContext",
        {
            InstanceMethod("contextSize", &LlamaContext::GetContextSize),
            InstanceMethod("stateSize", &LlamaContext::GetStateSize),
            InstanceMethod("eval", &LlamaContext::EvalSequence),
            InstanceMethod("removeTokens", &LlamaContext::RemoveTokens),
            InstanceMethod("shiftTokens", &LlamaContext::ShiftTokens),
            InstanceMethod("dispose", &LlamaContext::Dispose),
        });
    exports.Set("LlamaContext", def);
  }
};
