//
//  model.h
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

static Napi::Value getNapiToken(const Napi::CallbackInfo &info, llama_model *model, llama_token token)
{
  if (token < 0)
  {
    return Napi::Number::From(info.Env(), -1);
  }

  auto tokenAttributes = llama_token_get_attr(model, token);

  if (tokenAttributes & LLAMA_TOKEN_ATTR_UNDEFINED || tokenAttributes & LLAMA_TOKEN_ATTR_UNKNOWN)
  {
    return Napi::Number::From(info.Env(), -1);
  }

  return Napi::Number::From(info.Env(), token);
}

static Napi::Value getNapiControlToken(const Napi::CallbackInfo &info, llama_model *model, llama_token token)
{
  if (token < 0)
  {
    return Napi::Number::From(info.Env(), -1);
  }

  auto tokenAttributes = llama_token_get_attr(model, token);

  if (!(tokenAttributes & LLAMA_TOKEN_ATTR_CONTROL) && !(tokenAttributes & LLAMA_TOKEN_ATTR_UNDEFINED))
  {
    return Napi::Number::From(info.Env(), -1);
  }

  return Napi::Number::From(info.Env(), token);
}

class LlamaModel : public Napi::ObjectWrap<LlamaModel>
{
public:
  llama_model_params params;
  llama_model *model;

  std::string modelPath;
  Napi::Reference<Napi::Object> options;

  class LoaderWorker : public Napi::AsyncProgressQueueWorker<float>
  {
  public:
    LoaderWorker(
        Napi::Function &okCallback,
        Napi::Function &progressCallback,
        LlamaModel *model)
        : AsyncProgressQueueWorker(okCallback), model(model)
    {
      this->progressCallback.Reset(progressCallback, 1);
      model->Ref();
    }

    ~LoaderWorker()
    {
      model->Unref();
    }

  private:
    LlamaModel *model;
    Napi::FunctionReference progressCallback;
    bool aborted = false;

    const ExecutionProgress *progress;

    static bool
    progress_callback(float progress, void *user_data)
    {
      auto worker = (LoaderWorker *)user_data;
      worker->progress->Send(&progress, 1);
      return !worker->aborted;
    }

    void Execute(const ExecutionProgress &progress)
    {
      try
      {
        this->progress = &progress;
        model->params.progress_callback_user_data = this;
        model->params.progress_callback = progress_callback;
        model->model = llama_load_model_from_file(model->modelPath.c_str(), model->params);
      }
      catch (const std::exception &e)
      {
        SetError(e.what());
      }
      catch (...)
      {
        SetError("Failed to load model");
      }
    }
    void OnOK()
    {
      Napi::HandleScope scope(Env());
      if (model->model == NULL)
      {
        auto error = Napi::Error::New(Env(), "Failed to load model").Value();
        Callback().Call({error});
      }
      else
      {
        Callback().Call({});
      }
    }

    void OnError(const Napi::Error &err)
    {
      Napi::HandleScope scope(Env());
      Callback().Call({err.Value()});
    }
    void OnProgress(const float *data, size_t /* count */)
    {
      Napi::HandleScope scope(Env());
      if (!progressCallback.IsEmpty())
      {
        auto result = progressCallback.Call(Receiver().Value(), {Napi::Number::New(Env(), *data)});
        aborted = !result.ToBoolean();
      }
    }
  };

  LlamaModel(const Napi::CallbackInfo &info) : Napi::ObjectWrap<LlamaModel>(info)
  {
    params = llama_model_default_params();
    modelPath = info[0].As<Napi::String>().Utf8Value();
    options = Napi::Persistent(info[1].As<Napi::Object>());

    if (options.Value().Has("gpuLayers"))
    {
      params.n_gpu_layers = options.Value().Get("gpuLayers").As<Napi::Number>().Int32Value();
    }

    if (options.Value().Has("vocabOnly"))
    {
      params.vocab_only = options.Value().Get("vocabOnly").As<Napi::Boolean>().Value();
    }

    if (options.Value().Has("useMmap"))
    {
      params.use_mmap = options.Value().Get("useMmap").As<Napi::Boolean>().Value();
    }

    if (options.Value().Has("useMlock"))
    {
      params.use_mlock = options.Value().Get("useMlock").As<Napi::Boolean>().Value();
    }

    if (options.Value().Has("checkTensors"))
    {
      params.check_tensors = options.Value().Get("checkTensors").As<Napi::Boolean>().Value();
    }

    auto progress = options.Value().Get("onLoadProgress").As<Napi::Function>();
    auto complete = options.Value().Get("onComplete").As<Napi::Function>();
    auto worker = new LoaderWorker(complete, progress, this);
    worker->Queue();
  }

  ~LlamaModel()
  {
    dispose();
    options.Unref();
  }

  void dispose()
  {
    if (model == NULL)
    {
      return;
    }
    llama_free_model(model);
    model = NULL;
  }

  Napi::Value Dispose(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      return Env().Undefined();
    }
    dispose();
  }

  Napi::Value Tokenize(const Napi::CallbackInfo &info)
  {
    std::string text = info[0].As<Napi::String>().Utf8Value();
    bool encodeSpecial = info[1].As<Napi::Boolean>().Value();

    std::vector<llama_token> tokens = llama_tokenize(model, text, false, encodeSpecial);

    Napi::Uint32Array result = Napi::Uint32Array::New(Env(), tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i)
    {
      result[i] = static_cast<uint32_t>(tokens[i]);
    }

    return result;
  }
  Napi::Value Detokenize(const Napi::CallbackInfo &info)
  {
    Napi::Uint32Array tokens = info[0].As<Napi::Uint32Array>();
    bool decodeSpecial = info.Length() > 0
                             ? info[1].As<Napi::Boolean>().Value()
                             : false;

    std::vector<char> result(8, 0);
    const int n_length = llama_detokenize(model, (llama_token *)tokens.Data(), tokens.ElementLength(), result.data(), result.size(), false, decodeSpecial);

    if (n_length < 0)
    {
      result.resize(-n_length);
      int check = llama_detokenize(model, (llama_token *)tokens.Data(), tokens.ElementLength(), result.data(), result.size(), false, decodeSpecial);
      GGML_ASSERT(check == -n_length);
    }
    else
    {
      result.resize(n_length);
    }

    return Napi::String::New(Env(), result.data(), result.size());
  }

  Napi::Value ContextSize(const Napi::CallbackInfo &info)
  {
    return Napi::Number::From(Env(), llama_n_ctx_train(model));
  }

  Napi::Value EmbeddingSize(const Napi::CallbackInfo &info)
  {
    return Napi::Number::From(Env(), llama_n_embd(model));
  }

  Napi::Value TotalSize(const Napi::CallbackInfo &info)
  {
    return Napi::Number::From(Env(), llama_model_size(model));
  }

  Napi::Value TotalParameters(const Napi::CallbackInfo &info)
  {
    return Napi::Number::From(Env(), llama_model_n_params(model));
  }

  Napi::Value Description(const Napi::CallbackInfo &info)
  {
    char model_desc[128];
    int actual_length = llama_model_desc(model, model_desc, sizeof(model_desc));
    return Napi::String::New(Env(), model_desc, actual_length);
  }

  Napi::Value TokenBos(const Napi::CallbackInfo &info)
  {
    return getNapiControlToken(info, model, llama_token_bos(model));
  }
  Napi::Value TokenEos(const Napi::CallbackInfo &info)
  {
    return getNapiControlToken(info, model, llama_token_eos(model));
  }
  Napi::Value TokenNl(const Napi::CallbackInfo &info)
  {
    return getNapiToken(info, model, llama_token_nl(model));
  }
  Napi::Value PrefixToken(const Napi::CallbackInfo &info)
  {
    return getNapiControlToken(info, model, llama_token_prefix(model));
  }
  Napi::Value MiddleToken(const Napi::CallbackInfo &info)
  {
    return getNapiControlToken(info, model, llama_token_middle(model));
  }
  Napi::Value SuffixToken(const Napi::CallbackInfo &info)
  {
    return getNapiControlToken(info, model, llama_token_suffix(model));
  }
  Napi::Value EotToken(const Napi::CallbackInfo &info)
  {
    return getNapiControlToken(info, model, llama_token_eot(model));
  }
  Napi::Value TokenString(const Napi::CallbackInfo &info)
  {
    int token = info[0].As<Napi::Number>().Int32Value();
    const char *str = llama_token_get_text(model, token);
    return str == nullptr ? Env().Undefined() : Napi::String::New(Env(), str);
  }

  Napi::Value TokenAttributes(const Napi::CallbackInfo &info)
  {
    if (info[0].IsNumber() == false)
    {
      return Napi::Number::From(Env(), int32_t(LLAMA_TOKEN_ATTR_UNDEFINED));
    }

    int token = info[0].As<Napi::Number>().Int32Value();
    auto tokenAttributes = llama_token_get_attr(model, token);

    return Napi::Number::From(Env(), int32_t(tokenAttributes));
  }
  Napi::Value IsEogToken(const Napi::CallbackInfo &info)
  {
    if (info[0].IsNumber() == false)
    {
      return Napi::Boolean::New(Env(), false);
    }

    int token = info[0].As<Napi::Number>().Int32Value();

    return Napi::Boolean::New(Env(), llama_token_is_eog(model, token));
  }
  Napi::Value VocabularyType(const Napi::CallbackInfo &info)
  {
    auto vocabularyType = llama_vocab_type(model);

    return Napi::Number::From(Env(), int32_t(vocabularyType));
  }
  Napi::Value ShouldPrependBosToken(const Napi::CallbackInfo &info)
  {
    const int addBos = llama_add_bos_token(model);
    bool shouldPrependBos = addBos != -1 ? bool(addBos) : (llama_vocab_type(model) == LLAMA_VOCAB_TYPE_SPM);
    return Napi::Boolean::New(Env(), shouldPrependBos);
  }

  Napi::Value ModelSize(const Napi::CallbackInfo &info)
  {
    return Napi::Number::From(Env(), llama_model_size(model));
  }

  Napi::Value MetaLength(const Napi::CallbackInfo &info)
  {
    return Napi::Number::From(Env(), llama_model_meta_count(model));
  }

  Napi::Value MetaKey(const Napi::CallbackInfo &info)
  {
    int idx = info[0].As<Napi::Number>().Int32Value();
    auto size = llama_model_meta_key_by_index(model, idx, NULL, 0);
    std::vector<char> result(size + 1, 0);

    llama_model_meta_key_by_index(model, idx, result.data(), result.size());

    return Napi::String::New(Env(), result.data(), result.size() - 1);
  }

  Napi::Value MetaValue(const Napi::CallbackInfo &info)
  {
    int idx = info[0].As<Napi::Number>().Int32Value();
    auto size = llama_model_meta_val_str_by_index(model, idx, NULL, 0);
    std::vector<char> result(size + 1, 0);

    llama_model_meta_val_str_by_index(model, idx, result.data(), result.size());

    return Napi::String::New(Env(), result.data(), result.size() - 1);
  }

  Napi::Value ChatApplyTemplate(const Napi::CallbackInfo &info)
  {
    std::string tmpl = info[0].As<Napi::String>().Utf8Value();
    std::vector<llama_chat_msg> msgs;

    auto _msgs = info[1].As<Napi::Array>();
    for (size_t i = 0; i < _msgs.Length(); ++i)
    {
      auto role = _msgs.Get(i).As<Napi::Object>().Get("role").As<Napi::String>().Utf8Value();
      auto content = _msgs.Get(i).As<Napi::Object>().Get("content").As<Napi::String>().Utf8Value();
      msgs.push_back({role, content});
    }

    auto result = llama_chat_apply_template(model, tmpl, msgs, false);

    return Napi::String::New(Env(), result);
  }

  static void init(Napi::Object exports)
  {
    auto def = DefineClass(
        exports.Env(),
        "LlamaModel",
        {
            InstanceMethod("tokenize", &LlamaModel::Tokenize),
            InstanceMethod("detokenize", &LlamaModel::Detokenize),
            InstanceMethod("contextSize", &LlamaModel::ContextSize),
            InstanceMethod("embeddingSize", &LlamaModel::EmbeddingSize),
            InstanceMethod("totalSize", &LlamaModel::TotalSize),
            InstanceMethod("totalParameters", &LlamaModel::TotalParameters),
            InstanceMethod("description", &LlamaModel::Description),
            InstanceMethod("tokenBos", &LlamaModel::TokenBos),
            InstanceMethod("tokenEos", &LlamaModel::TokenEos),
            InstanceMethod("tokenNl", &LlamaModel::TokenNl),
            InstanceMethod("prefixToken", &LlamaModel::PrefixToken),
            InstanceMethod("middleToken", &LlamaModel::MiddleToken),
            InstanceMethod("suffixToken", &LlamaModel::SuffixToken),
            InstanceMethod("eotToken", &LlamaModel::EotToken),
            InstanceMethod("tokenString", &LlamaModel::TokenString),
            InstanceMethod("tokenAttributes", &LlamaModel::TokenAttributes),
            InstanceMethod("isEogToken", &LlamaModel::IsEogToken),
            InstanceMethod("vocabularyType", &LlamaModel::VocabularyType),
            InstanceMethod("shouldPrependBosToken", &LlamaModel::ShouldPrependBosToken),
            InstanceMethod("modelSize", &LlamaModel::ModelSize),
            InstanceMethod("metaLength", &LlamaModel::MetaLength),
            InstanceMethod("metaKey", &LlamaModel::MetaKey),
            InstanceMethod("metaValue", &LlamaModel::MetaValue),
            InstanceMethod("chatApplyTemplate", &LlamaModel::ChatApplyTemplate),
            InstanceMethod("dispose", &LlamaModel::Dispose),
        });
    exports.Set("LlamaModel", def);
  }
};
