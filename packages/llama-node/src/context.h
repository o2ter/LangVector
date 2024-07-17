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
#include "worker.h"

class LlamaContextSampleCandidates : public Napi::ObjectWrap<LlamaContextSampleCandidates>
{
public:
  std::vector<llama_token_data> candidates;

  LlamaContextSampleCandidates(const Napi::CallbackInfo &info);

  Napi::Value IsValid(const Napi::CallbackInfo &info)
  {
    for (auto candidate : candidates)
    {
      if (candidate.logit != -INFINITY)
      {
        return Napi::Boolean::New(info.Env(), true);
      }
    }
    return Napi::Boolean::New(info.Env(), false);
  }

  static void init(Napi::Object exports)
  {
    auto def = DefineClass(
        exports.Env(),
        "LlamaContextSampleCandidates",
        {
            InstanceMethod("isValid", &LlamaContextSampleCandidates::IsValid),
        });
    exports.Set("LlamaContextSampleCandidates", def);
  }
};

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

    if (options.Has("seed"))
    {
      params.seed = options.Get("seed").As<Napi::Number>().Uint32Value();
    }

    if (options.Has("contextSize"))
    {
      params.n_ctx = options.Get("contextSize").As<Napi::Number>().Uint32Value();
    }

    if (options.Has("batchSize"))
    {
      params.n_batch = options.Get("batchSize").As<Napi::Number>().Uint32Value();
      params.n_ubatch = params.n_batch;
    }

    if (options.Has("flashAttention"))
    {
      params.flash_attn = options.Get("flashAttention").As<Napi::Boolean>().Value();
    }

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

  Napi::Value GetBatchSize(const Napi::CallbackInfo &info)
  {
    return Napi::Number::From(Env(), llama_n_batch(ctx));
  }

  Napi::Value GetStateSize(const Napi::CallbackInfo &info)
  {
    return Napi::Number::From(Env(), llama_state_get_size(ctx));
  }

  Napi::Value EvalSequence(const Napi::CallbackInfo &info)
  {
    Napi::Uint32Array tokens = info[0].As<Napi::Uint32Array>();
    int32_t startPos = info[1].As<Napi::Number>().Int32Value();
    bool logitEnd = info[2].As<Napi::Boolean>().Value();

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
            llama_batch_add(batch, tokens[i], i + startPos, {0}, logitEnd && i + 1 == token_length);
          }
          if (llama_decode(ctx, batch) < 0)
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

  Napi::Value SampleToken(const Napi::CallbackInfo &info)
  {
    float temperature = 0.0f;
    float min_p = 0;
    int32_t top_k = 40;
    float top_p = 0.95f;

    auto candidates = Napi::ObjectWrap<LlamaContextSampleCandidates>::Unwrap(info[0].As<Napi::Object>());
    candidates->Ref();

    Napi::Object options = info[1].As<Napi::Object>();

    if (options.Has("temperature"))
    {
      temperature = options.Get("temperature").As<Napi::Number>().FloatValue();
    }

    if (options.Has("minP"))
    {
      min_p = options.Get("minP").As<Napi::Number>().FloatValue();
    }

    if (options.Has("topK"))
    {
      top_k = options.Get("topK").As<Napi::Number>().Int32Value();
    }

    if (options.Has("topP"))
    {
      top_p = options.Get("topP").As<Napi::Number>().FloatValue();
    }

    this->Ref();

    auto worker = new _AsyncWorkerWithResult<llama_token>(
        Env(),
        [=]()
        {
          llama_token result;
          llama_token_data_array candidates_p = {candidates->candidates.data(), candidates->candidates.size(), false};

          if (temperature <= 0)
          {
            result = llama_sample_token_greedy(ctx, &candidates_p);
          }
          else
          {
            const int32_t resolved_top_k =
                top_k <= 0 ? llama_n_vocab(model->model) : std::min(top_k, llama_n_vocab(model->model));
            const int32_t n_probs = 0;          // Number of probabilities to keep - 0 = disabled
            const float tfs_z = 1.00f;          // Tail free sampling - 1.0 = disabled
            const float typical_p = 1.00f;      // Typical probability - 1.0 = disabled
            const float resolved_top_p = top_p; // Top p sampling - 1.0 = disabled

            // Temperature sampling
            size_t min_keep = std::max(1, n_probs);
            llama_sample_top_k(ctx, &candidates_p, resolved_top_k, min_keep);
            llama_sample_tail_free(ctx, &candidates_p, tfs_z, min_keep);
            llama_sample_typical(ctx, &candidates_p, typical_p, min_keep);
            llama_sample_top_p(ctx, &candidates_p, resolved_top_p, min_keep);
            llama_sample_min_p(ctx, &candidates_p, min_p, min_keep);
            llama_sample_temp(ctx, &candidates_p, temperature);
            result = llama_sample_token(ctx, &candidates_p);
          }

          return result;
        },
        [=](Napi::Env env, llama_token result)
        {
          return Napi::Number::New(Env(), result);
        },
        [=]()
        {
          this->Unref();
          candidates->Unref();
        });

    worker->Queue();
    return worker->Promise();
  }

  Napi::Value RemoveTokens(const Napi::CallbackInfo &info)
  {
    int32_t startPos = info[0].As<Napi::Number>().Int32Value();
    int32_t endPos = info[1].As<Napi::Number>().Int32Value();

    bool result = llama_kv_cache_seq_rm(ctx, 0, startPos, endPos);

    return Napi::Boolean::New(Env(), result);
  }
  Napi::Value ShiftTokens(const Napi::CallbackInfo &info)
  {
    int32_t startPos = info[0].As<Napi::Number>().Int32Value();
    int32_t endPos = info[1].As<Napi::Number>().Int32Value();
    int32_t shiftDelta = info[2].As<Napi::Number>().Int32Value();

    llama_kv_cache_seq_add(ctx, 0, startPos, endPos, shiftDelta);

    return Env().Undefined();
  }
  static void init(Napi::Object exports)
  {
    auto def = DefineClass(
        exports.Env(),
        "LlamaContext",
        {
            InstanceMethod("contextSize", &LlamaContext::GetContextSize),
            InstanceMethod("batchSize", &LlamaContext::GetBatchSize),
            InstanceMethod("stateSize", &LlamaContext::GetStateSize),
            InstanceMethod("eval", &LlamaContext::EvalSequence),
            InstanceMethod("sampleToken", &LlamaContext::SampleToken),
            InstanceMethod("removeTokens", &LlamaContext::RemoveTokens),
            InstanceMethod("shiftTokens", &LlamaContext::ShiftTokens),
            InstanceMethod("dispose", &LlamaContext::Dispose),
        });
    exports.Set("LlamaContext", def);
  }
};

LlamaContextSampleCandidates::LlamaContextSampleCandidates(const Napi::CallbackInfo &info) : Napi::ObjectWrap<LlamaContextSampleCandidates>(info)
{
  float repeat_penalty = 1.10f;                   // 1.0 = disabled
  float repeat_penalty_presence_penalty = 0.00f;  // 0.0 = disabled
  float repeat_penalty_frequency_penalty = 0.00f; // 0.0 = disabled
  std::unordered_map<llama_token, float> tokenBias;
  std::vector<llama_token> repeat_penalty_tokens;

  auto ctx = Napi::ObjectWrap<LlamaContext>::Unwrap(info[0].As<Napi::Object>());
  Napi::Object options = info[1].As<Napi::Object>();

  if (options.Has("tokenBias"))
  {
    auto _tokenBias = options.Get("tokenBias").As<Napi::Array>();
    for (size_t i = 0; i < _tokenBias.Length(); ++i)
    {
      auto key = _tokenBias.Get(i).As<Napi::Object>().Get("key").As<Napi::Number>().Uint32Value();
      auto value = _tokenBias.Get(i).As<Napi::Object>().Get("value").As<Napi::Number>().FloatValue();
      tokenBias[static_cast<llama_token>(key)] = value;
    }
  }

  if (options.Has("repeatPenalty"))
  {
    repeat_penalty = options.Get("repeatPenalty").As<Napi::Number>().FloatValue();
  }

  if (options.Has("repeatPenaltyTokens"))
  {
    Napi::Uint32Array array = options.Get("repeatPenaltyTokens").As<Napi::Uint32Array>();
    repeat_penalty_tokens.reserve(array.ElementLength());
    for (size_t i = 0; i < array.ElementLength(); ++i)
    {
      repeat_penalty_tokens.push_back(static_cast<llama_token>(array[i]));
    }
  }

  if (options.Has("repeatPenaltyPresencePenalty"))
  {
    repeat_penalty_presence_penalty = options.Get("repeatPenaltyPresencePenalty").As<Napi::Number>().FloatValue();
  }

  if (options.Has("repeatPenaltyFrequencyPenalty"))
  {
    repeat_penalty_frequency_penalty = options.Get("repeatPenaltyFrequencyPenalty").As<Napi::Number>().FloatValue();
  }

  // Select the best prediction.
  if (llama_get_logits(ctx->ctx) == nullptr)
  {
    throw std::runtime_error("This model does not support token generation");
  }

  auto logits = llama_get_logits_ith(ctx->ctx, -1);
  auto n_vocab = llama_n_vocab(ctx->model->model);

  candidates.reserve(n_vocab);

  for (llama_token token_id = 0; token_id < n_vocab; ++token_id)
  {
    auto logit = logits[token_id];
    if (!tokenBias.empty() && tokenBias.find(token_id) != tokenBias.end())
    {
      auto logitBias = tokenBias.at(token_id);
      if (logitBias == -INFINITY || logitBias < -INFINITY)
      {
        if (!llama_token_is_eog(ctx->model->model, token_id))
        {
          logit = -INFINITY;
        }
      }
      else
      {
        logit += logitBias;
      }
    }
    candidates.emplace_back(llama_token_data{token_id, logit, 0.0f});
  }

  llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

  if (!repeat_penalty_tokens.empty())
  {
    llama_sample_repetition_penalties(
        ctx->ctx,
        &candidates_p,
        repeat_penalty_tokens.data(),
        repeat_penalty_tokens.size(),
        repeat_penalty,
        repeat_penalty_frequency_penalty,
        repeat_penalty_presence_penalty);
  }
}
