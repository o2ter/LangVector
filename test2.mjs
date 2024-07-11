
import _ from 'lodash';
import { LlamaDevice } from './dist/index.mjs';

console.log(LlamaDevice.systemInfo());
console.log(LlamaDevice.getSupportsGpuOffloading());
console.log(LlamaDevice.getSupportsMmap());
console.log(LlamaDevice.getSupportsMlock());
console.log(LlamaDevice.getBlockSizeForGgmlType(0));
console.log(LlamaDevice.getTypeSizeForGgmlType(0));
console.log(LlamaDevice.getConsts());
console.log(LlamaDevice.getGpuType());
console.log(LlamaDevice.getGpuDeviceInfo());
console.log(LlamaDevice.getGpuVramInfo());
