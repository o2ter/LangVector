
import _ from 'lodash';
import { LlamaDevice } from '../dist/index.mjs';

console.log(LlamaDevice.systemInfo());
console.log(LlamaDevice.supportsGpuOffloading());
console.log(LlamaDevice.supportsMmap());
console.log(LlamaDevice.supportsMlock());
console.log(LlamaDevice.blockSizeForGgmlType(0));
console.log(LlamaDevice.typeSizeForGgmlType(0));
console.log(LlamaDevice.consts());
console.log(LlamaDevice.gpuType());
console.log(LlamaDevice.gpuDeviceInfo());
console.log(LlamaDevice.gpuVramInfo());
