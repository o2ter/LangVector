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
  llama_model_params model_params;
  llama_model *model;

  std::string modelPath;
  Napi::Reference<Napi::Object> options;

  LlamaModel(const Napi::CallbackInfo &info) : Napi::ObjectWrap<LlamaModel>(info)
  {
    model_params = llama_model_default_params();
    modelPath = info[0].As<Napi::String>().Utf8Value();
    options = Napi::Persistent(info[1].As<Napi::Object>());

    if (options.Value().Has("gpuLayers"))
    {
      model_params.n_gpu_layers = options.Value().Get("gpuLayers").As<Napi::Number>().Int32Value();
    }

    if (options.Value().Has("vocabOnly"))
    {
      model_params.vocab_only = options.Value().Get("vocabOnly").As<Napi::Boolean>().Value();
    }

    if (options.Value().Has("useMmap"))
    {
      model_params.use_mmap = options.Value().Get("useMmap").As<Napi::Boolean>().Value();
    }

    if (options.Value().Has("useMlock"))
    {
      model_params.use_mlock = options.Value().Get("useMlock").As<Napi::Boolean>().Value();
    }

    if (options.Value().Has("checkTensors"))
    {
      model_params.check_tensors = options.Value().Get("checkTensors").As<Napi::Boolean>().Value();
    }

    auto complete = ThreadSafeCallback(
        info,
        [this](const Napi::CallbackInfo &info)
        {
          if (options.Value().Has("onComplete"))
          {
            auto callback = options.Value().Get("onComplete").As<Napi::Function>();
            if (callback.IsFunction())
            {
              if (model == NULL)
              {
                auto error = Napi::Error::New(info.Env(), "Failed to load model").Value();
                callback.Call({error});
              }
              else
              {
                callback.Call({});
              }
            }
          }
        });

    ThreadPool::excute(
        [this, complete]()
        {
          model = llama_load_model_from_file(modelPath.c_str(), model_params);
          complete.BlockingCall();
          complete.Release();
        });
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
      return info.Env().Undefined();
    }
    dispose();
  }

  Napi::Value Tokenize(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    std::string text = info[0].As<Napi::String>().Utf8Value();
    bool encodeSpecial = info[1].As<Napi::Boolean>().Value();

    std::vector<llama_token> tokens = llama_tokenize(model, text, false, encodeSpecial);

    Napi::Uint32Array result = Napi::Uint32Array::New(info.Env(), tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i)
    {
      result[i] = static_cast<uint32_t>(tokens[i]);
    }

    return result;
  }
  Napi::Value Detokenize(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

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

    return Napi::String::New(info.Env(), result.data(), result.size());
  }

  Napi::Value ContextSize(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    return Napi::Number::From(info.Env(), llama_n_ctx_train(model));
  }

  Napi::Value EmbeddingSize(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    return Napi::Number::From(info.Env(), llama_n_embd(model));
  }

  Napi::Value TotalSize(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    return Napi::Number::From(info.Env(), llama_model_size(model));
  }

  Napi::Value TotalParameters(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    return Napi::Number::From(info.Env(), llama_model_n_params(model));
  }

  Napi::Value Description(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    char model_desc[128];
    int actual_length = llama_model_desc(model, model_desc, sizeof(model_desc));

    return Napi::String::New(info.Env(), model_desc, actual_length);
  }

  Napi::Value TokenBos(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    return getNapiControlToken(info, model, llama_token_bos(model));
  }
  Napi::Value TokenEos(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    return getNapiControlToken(info, model, llama_token_eos(model));
  }
  Napi::Value TokenNl(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    return getNapiToken(info, model, llama_token_nl(model));
  }
  Napi::Value PrefixToken(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    return getNapiControlToken(info, model, llama_token_prefix(model));
  }
  Napi::Value MiddleToken(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    return getNapiControlToken(info, model, llama_token_middle(model));
  }
  Napi::Value SuffixToken(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    return getNapiControlToken(info, model, llama_token_suffix(model));
  }
  Napi::Value EotToken(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    return getNapiControlToken(info, model, llama_token_eot(model));
  }
  Napi::Value TokenString(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    int token = info[0].As<Napi::Number>().Int32Value();
    std::stringstream ss;

    const char *str = llama_token_get_text(model, token);
    if (str == nullptr)
    {
      return info.Env().Undefined();
    }

    ss << str;

    return Napi::String::New(info.Env(), ss.str());
  }

  Napi::Value TokenAttributes(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    if (info[0].IsNumber() == false)
    {
      return Napi::Number::From(info.Env(), int32_t(LLAMA_TOKEN_ATTR_UNDEFINED));
    }

    int token = info[0].As<Napi::Number>().Int32Value();
    auto tokenAttributes = llama_token_get_attr(model, token);

    return Napi::Number::From(info.Env(), int32_t(tokenAttributes));
  }
  Napi::Value IsEogToken(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    if (info[0].IsNumber() == false)
    {
      return Napi::Boolean::New(info.Env(), false);
    }

    int token = info[0].As<Napi::Number>().Int32Value();

    return Napi::Boolean::New(info.Env(), llama_token_is_eog(model, token));
  }
  Napi::Value VocabularyType(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    auto vocabularyType = llama_vocab_type(model);

    return Napi::Number::From(info.Env(), int32_t(vocabularyType));
  }
  Napi::Value ShouldPrependBosToken(const Napi::CallbackInfo &info)
  {
    const int addBos = llama_add_bos_token(model);

    bool shouldPrependBos = addBos != -1 ? bool(addBos) : (llama_vocab_type(model) == LLAMA_VOCAB_TYPE_SPM);

    return Napi::Boolean::New(info.Env(), shouldPrependBos);
  }

  Napi::Value ModelSize(const Napi::CallbackInfo &info)
  {
    return Napi::Number::From(info.Env(), llama_model_size(model));
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
            InstanceMethod("dispose", &LlamaModel::Dispose),
        });
    exports.Set("LlamaModel", def);
  }
};
