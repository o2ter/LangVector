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

#include <stddef.h>

#include "llama.h"
#include "common/common.h"
#include "napi.h"

#ifdef GPU_INFO_USE_CUDA
#include "gpuInfo/cuda-gpu-info.h"
#endif
#ifdef GPU_INFO_USE_VULKAN
#include "gpuInfo/vulkan-gpu-info.h"
#endif
#ifdef GPU_INFO_USE_METAL
#include "gpuInfo/metal-gpu-info.h"
#endif

Napi::Value getGpuVramInfo(const Napi::CallbackInfo &info)
{
  uint64_t total = 0;
  uint64_t used = 0;

#ifdef GPU_INFO_USE_CUDA
  size_t cudaDeviceTotal = 0;
  size_t cudaDeviceUsed = 0;
  bool cudeGetInfoSuccess = gpuInfoGetTotalCudaDevicesInfo(&cudaDeviceTotal, &cudaDeviceUsed, logCudaError);

  if (cudeGetInfoSuccess)
  {
    total += cudaDeviceTotal;
    used += cudaDeviceUsed;
  }
#endif

#ifdef GPU_INFO_USE_VULKAN
  uint64_t vulkanDeviceTotal = 0;
  uint64_t vulkanDeviceUsed = 0;
  const bool vulkanDeviceSupportsMemoryBudgetExtension = gpuInfoGetTotalVulkanDevicesInfo(&vulkanDeviceTotal, &vulkanDeviceUsed, logVulkanWarning);

  if (vulkanDeviceSupportsMemoryBudgetExtension)
  {
    total += vulkanDeviceTotal;
    used += vulkanDeviceUsed;
  }
#endif

#ifdef GPU_INFO_USE_METAL
  uint64_t metalDeviceTotal = 0;
  uint64_t metalDeviceUsed = 0;
  getMetalGpuInfo(&metalDeviceTotal, &metalDeviceUsed);

  total += metalDeviceTotal;
  used += metalDeviceUsed;
#endif

  Napi::Object result = Napi::Object::New(info.Env());
  result.Set("total", Napi::Number::From(info.Env(), total));
  result.Set("used", Napi::Number::From(info.Env(), used));

  return result;
}

Napi::Value getGpuDeviceInfo(const Napi::CallbackInfo &info)
{
  std::vector<std::string> deviceNames;

#ifdef GPU_INFO_USE_CUDA
  gpuInfoGetCudaDeviceNames(&deviceNames, logCudaError);
#endif

#ifdef GPU_INFO_USE_VULKAN
  gpuInfoGetVulkanDeviceNames(&deviceNames, logVulkanWarning);
#endif

#ifdef GPU_INFO_USE_METAL
  getMetalGpuDeviceNames(&deviceNames);
#endif

  Napi::Object result = Napi::Object::New(info.Env());

  Napi::Array deviceNamesNapiArray = Napi::Array::New(info.Env(), deviceNames.size());
  for (size_t i = 0; i < deviceNames.size(); ++i)
  {
    deviceNamesNapiArray[i] = Napi::String::New(info.Env(), deviceNames[i]);
  }
  result.Set("deviceNames", deviceNamesNapiArray);

  return result;
}

Napi::Value getGpuType(const Napi::CallbackInfo &info)
{
#ifdef GPU_INFO_USE_CUDA
  return Napi::String::New(info.Env(), "cuda");
#endif

#ifdef GPU_INFO_USE_VULKAN
  return Napi::String::New(info.Env(), "vulkan");
#endif

#ifdef GPU_INFO_USE_METAL
  return Napi::String::New(info.Env(), "metal");
#endif

  return info.Env().Undefined();
}

Napi::Value systemInfo(const Napi::CallbackInfo &info)
{
  return Napi::String::From(info.Env(), llama_print_system_info());
}

Napi::Value getSupportsGpuOffloading(const Napi::CallbackInfo &info)
{
  return Napi::Boolean::New(info.Env(), llama_supports_gpu_offload());
}

Napi::Value getSupportsMmap(const Napi::CallbackInfo &info)
{
  return Napi::Boolean::New(info.Env(), llama_supports_mmap());
}

Napi::Value getSupportsMlock(const Napi::CallbackInfo &info)
{
  return Napi::Boolean::New(info.Env(), llama_supports_mlock());
}

Napi::Value getBlockSizeForGgmlType(const Napi::CallbackInfo &info)
{
  const int ggmlType = info[0].As<Napi::Number>().Int32Value();

  if (ggmlType < 0 || ggmlType > GGML_TYPE_COUNT)
  {
    return info.Env().Undefined();
  }

  const auto blockSize = ggml_blck_size(static_cast<ggml_type>(ggmlType));

  return Napi::Number::New(info.Env(), blockSize);
}

Napi::Value getTypeSizeForGgmlType(const Napi::CallbackInfo &info)
{
  const int ggmlType = info[0].As<Napi::Number>().Int32Value();

  if (ggmlType < 0 || ggmlType > GGML_TYPE_COUNT)
  {
    return info.Env().Undefined();
  }

  const auto typeSize = ggml_type_size(static_cast<ggml_type>(ggmlType));

  return Napi::Number::New(info.Env(), typeSize);
}

Napi::Value getConsts(const Napi::CallbackInfo &info)
{
  Napi::Object consts = Napi::Object::New(info.Env());
  consts.Set("ggmlMaxDims", Napi::Number::New(info.Env(), GGML_MAX_DIMS));
  consts.Set("ggmlTypeF16Size", Napi::Number::New(info.Env(), ggml_type_size(GGML_TYPE_F16)));
  consts.Set("ggmlTypeF32Size", Napi::Number::New(info.Env(), ggml_type_size(GGML_TYPE_F32)));
  consts.Set("ggmlTensorOverhead", Napi::Number::New(info.Env(), ggml_tensor_overhead()));
  consts.Set("llamaMaxRngState", Napi::Number::New(info.Env(), LLAMA_MAX_RNG_STATE));
  consts.Set("llamaPosSize", Napi::Number::New(info.Env(), sizeof(llama_pos)));
  consts.Set("llamaSeqIdSize", Napi::Number::New(info.Env(), sizeof(llama_seq_id)));

  return consts;
}

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

  LlamaModel(const Napi::CallbackInfo &info) : Napi::ObjectWrap<LlamaModel>(info)
  {
    model_params = llama_model_default_params();
    modelPath = info[0].As<Napi::String>().Utf8Value();

    Napi::Object options = info[1].As<Napi::Object>();
    OnLoadProgressUserData *userData = NULL;

    if (options.Has("gpuLayers"))
    {
      model_params.n_gpu_layers = options.Get("gpuLayers").As<Napi::Number>().Int32Value();
    }

    if (options.Has("vocabOnly"))
    {
      model_params.vocab_only = options.Get("vocabOnly").As<Napi::Boolean>().Value();
    }

    if (options.Has("useMmap"))
    {
      model_params.use_mmap = options.Get("useMmap").As<Napi::Boolean>().Value();
    }

    if (options.Has("useMlock"))
    {
      model_params.use_mlock = options.Get("useMlock").As<Napi::Boolean>().Value();
    }

    if (options.Has("checkTensors"))
    {
      model_params.check_tensors = options.Get("checkTensors").As<Napi::Boolean>().Value();
    }

    if (options.Has("onLoadProgress"))
    {
      auto callback = options.Get("onLoadProgress").As<Napi::Function>();
      if (callback.IsFunction())
      {
        userData = new OnLoadProgressUserData{
          env : info.Env(),
          callback,
        };
        model_params.progress_callback_user_data = userData;
        model_params.progress_callback = OnLoadProgressCallback;
      }
    }

    model = llama_load_model_from_file(modelPath.c_str(), model_params);

    if (userData != NULL)
    {
      delete userData;
    }

    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Failed to load model").ThrowAsJavaScriptException();
      return;
    }
  }

  ~LlamaModel()
  {
    dispose();
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

  struct OnLoadProgressUserData
  {
    Napi::Env env;
    Napi::Function callback;
  };

  static bool OnLoadProgressCallback(float progress, void *user_data)
  {
    OnLoadProgressUserData *data = (OnLoadProgressUserData *)user_data;
    auto result = data->callback.Call({Napi::Number::New(data->env, progress)});
    return result.As<Napi::Boolean>().Value();
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
    bool specialTokens = info[1].As<Napi::Boolean>().Value();

    std::vector<llama_token> tokens = llama_tokenize(model, text, false, specialTokens);

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
    bool decodeSpecialTokens = info.Length() > 0
                                   ? info[1].As<Napi::Boolean>().Value()
                                   : false;

    std::vector<char> result(8, 0);
    const int n_length = llama_detokenize(model, (llama_token *)tokens.Data(), tokens.ElementLength(), result.data(), result.size(), false, decodeSpecialTokens);

    if (n_length < 0)
    {
      result.resize(-n_length);
      int check = llama_detokenize(model, (llama_token *)tokens.Data(), tokens.ElementLength(), result.data(), result.size(), false, decodeSpecialTokens);
      GGML_ASSERT(check == -n_length);
    }
    else
    {
      result.resize(n_length);
    }

    return Napi::String::New(info.Env(), result.data(), result.size());
  }

  Napi::Value GetTrainContextSize(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    return Napi::Number::From(info.Env(), llama_n_ctx_train(model));
  }

  Napi::Value GetEmbeddingVectorSize(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    return Napi::Number::From(info.Env(), llama_n_embd(model));
  }

  Napi::Value GetTotalSize(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    return Napi::Number::From(info.Env(), llama_model_size(model));
  }

  Napi::Value GetTotalParameters(const Napi::CallbackInfo &info)
  {
    if (model == NULL)
    {
      Napi::Error::New(info.Env(), "Model is disposed").ThrowAsJavaScriptException();
      return info.Env().Undefined();
    }

    return Napi::Number::From(info.Env(), llama_model_n_params(model));
  }

  Napi::Value GetModelDescription(const Napi::CallbackInfo &info)
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
  Napi::Value GetTokenString(const Napi::CallbackInfo &info)
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

  Napi::Value GetTokenAttributes(const Napi::CallbackInfo &info)
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
  Napi::Value GetVocabularyType(const Napi::CallbackInfo &info)
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

  Napi::Value GetModelSize(const Napi::CallbackInfo &info)
  {
    return Napi::Number::From(info.Env(), llama_model_size(model));
  }

  static void init(Napi::Object exports)
  {
    exports.Set(
        "LlamaModel",
        DefineClass(
            exports.Env(),
            "LlamaModel",
            {
                InstanceMethod("tokenize", &LlamaModel::Tokenize),
                InstanceMethod("detokenize", &LlamaModel::Detokenize),
                InstanceMethod("getTrainContextSize", &LlamaModel::GetTrainContextSize),
                InstanceMethod("getEmbeddingVectorSize", &LlamaModel::GetEmbeddingVectorSize),
                InstanceMethod("getTotalSize", &LlamaModel::GetTotalSize),
                InstanceMethod("getTotalParameters", &LlamaModel::GetTotalParameters),
                InstanceMethod("getModelDescription", &LlamaModel::GetModelDescription),
                InstanceMethod("tokenBos", &LlamaModel::TokenBos),
                InstanceMethod("tokenEos", &LlamaModel::TokenEos),
                InstanceMethod("tokenNl", &LlamaModel::TokenNl),
                InstanceMethod("prefixToken", &LlamaModel::PrefixToken),
                InstanceMethod("middleToken", &LlamaModel::MiddleToken),
                InstanceMethod("suffixToken", &LlamaModel::SuffixToken),
                InstanceMethod("eotToken", &LlamaModel::EotToken),
                InstanceMethod("getTokenString", &LlamaModel::GetTokenString),
                InstanceMethod("getTokenAttributes", &LlamaModel::GetTokenAttributes),
                InstanceMethod("isEogToken", &LlamaModel::IsEogToken),
                InstanceMethod("getVocabularyType", &LlamaModel::GetVocabularyType),
                InstanceMethod("shouldPrependBosToken", &LlamaModel::ShouldPrependBosToken),
                InstanceMethod("getModelSize", &LlamaModel::GetModelSize),
                InstanceMethod("dispose", &LlamaModel::Dispose),
            }));
  }
};

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
  return exports;
}

NODE_API_MODULE(NODE_GYP_MODULE_NAME, registerCallback)
