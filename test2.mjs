
import _ from 'lodash';
import { llamaCpp } from './dist/index.mjs';

console.log(llamaCpp.systemInfo());
console.log(llamaCpp.getGpuType());
console.log(llamaCpp.getGpuDeviceInfo());
console.log(llamaCpp.getGpuVramInfo());

