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

Napi::Object registerCallback(Napi::Env env, Napi::Object exports)
{
  llama_backend_init();
  exports.DefineProperties({
      Napi::PropertyDescriptor::Function("systemInfo", systemInfo),
      Napi::PropertyDescriptor::Function("getGpuVramInfo", getGpuVramInfo),
      Napi::PropertyDescriptor::Function("getGpuDeviceInfo", getGpuDeviceInfo),
      Napi::PropertyDescriptor::Function("getGpuType", getGpuType),
  });
  return exports;
}

NODE_API_MODULE(NODE_GYP_MODULE_NAME, registerCallback)
