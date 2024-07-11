#include <stddef.h>

#include "llama.h"
#include "napi.h"

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

Napi::Object registerCallback(Napi::Env env, Napi::Object exports)
{
  exports.DefineProperties({
      Napi::PropertyDescriptor::Function("getGpuType", getGpuType),
  });

  return exports;
}

NODE_API_MODULE(NODE_GYP_MODULE_NAME, registerCallback)
