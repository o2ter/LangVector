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

Napi::Value systemInfo(const Napi::CallbackInfo &info) { return Napi::String::From(info.Env(), llama_print_system_info()); }

Napi::Object registerCallback(Napi::Env env, Napi::Object exports)
{
  llama_backend_init();
  exports.DefineProperties({
      Napi::PropertyDescriptor::Function("systemInfo", systemInfo),
      Napi::PropertyDescriptor::Function("getGpuType", getGpuType),
  });
  return exports;
}

NODE_API_MODULE(NODE_GYP_MODULE_NAME, registerCallback)
