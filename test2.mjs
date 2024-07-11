
import _ from 'lodash';
import { LlamaDevice } from './dist/index.mjs';

console.log(LlamaDevice.systemInfo());
console.log(LlamaDevice.getGpuType());
console.log(LlamaDevice.getGpuDeviceInfo());
console.log(LlamaDevice.getGpuVramInfo());

