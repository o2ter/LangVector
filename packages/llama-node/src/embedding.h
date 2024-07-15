//
//  embedding.h
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

class LlamaEmbeddingContext : public Napi::ObjectWrap<LlamaEmbeddingContext>
{
public:
  LlamaModel *model;
  llama_context_params params;
  llama_context *ctx;

  LlamaEmbeddingContext(const Napi::CallbackInfo &info) : Napi::ObjectWrap<LlamaEmbeddingContext>(info)
  {
    model = Napi::ObjectWrap<LlamaModel>::Unwrap(info[0].As<Napi::Object>());
    model->Ref();

    auto hardware_concurrency = std::thread::hardware_concurrency();

    params = llama_context_default_params();
    params.n_ctx = 0;
    params.n_seq_max = 1;
    params.n_threads = hardware_concurrency;
    params.n_threads_batch = hardware_concurrency;
    params.embeddings = true;

    Napi::Object options = info[1].As<Napi::Object>();

    params.n_batch = options.Get("batchSize").As<Napi::Number>().Uint32Value();
    params.n_ubatch = params.n_batch;
    params.n_ctx = params.n_batch;

    if (options.Has("threads"))
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

  ~LlamaEmbeddingContext()
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

  Napi::Value EvalEmbedding(const Napi::CallbackInfo &info)
  {
    Napi::Uint32Array tokens = info[0].As<Napi::Uint32Array>();

    this->Ref();

    auto worker = new _AsyncWorker(
        Env(),
        [=]()
        {
          auto token_length = tokens.ElementLength();
          size_t n_batch = llama_n_batch(ctx);

          if (token_length > n_batch) {
            throw std::runtime_error("Number of tokens exceeds batch size");
          }

          llama_batch batch = llama_batch_init(token_length, 0, 1);

          for (size_t i = 0; i < token_length; ++i)
          {
            llama_batch_add(batch, tokens[i], i, {0}, i + 1 == token_length);
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
  Napi::Value GetEmbedding(const Napi::CallbackInfo &info)
  {
    const int n_embd = llama_n_embd(model->model);
    const auto *embeddings = llama_get_embeddings_seq(ctx, 0);
    if (embeddings == NULL)
    {
      embeddings = llama_get_embeddings_ith(ctx, -1);

      if (embeddings == NULL)
      {
        Napi::Error::New(Env(), "Failed to get embeddings").ThrowAsJavaScriptException();
        return Env().Undefined();
      }
    }

    Napi::Float64Array result = Napi::Float64Array::New(Env(), n_embd);
    for (size_t i = 0; i < n_embd; ++i)
    {
      result[i] = embeddings[i];
    }

    return result;
  }

  static void init(Napi::Object exports)
  {
    auto def = DefineClass(
        exports.Env(),
        "LlamaEmbeddingContext",
        {
            InstanceMethod("eval", &LlamaEmbeddingContext::EvalEmbedding),
            InstanceMethod("embedding", &LlamaEmbeddingContext::GetEmbedding),
            InstanceMethod("dispose", &LlamaEmbeddingContext::Dispose),
        });
    exports.Set("LlamaEmbeddingContext", def);
  }
};
