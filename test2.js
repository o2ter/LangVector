const path = require('path');
const { LlamaDevice } = require('./dist/index.js');

const modelsDir = path.join(__dirname, './models');

(async () => {

  const model = await LlamaDevice.loadModel({
    modelPath: path.join(modelsDir, 'meta-llama/Meta-Llama-3-8B-Instruct/ggml-model-q3_k_m.gguf'),
  });

  const context = model.createContext();

  const messages = [
    'hi',
  ];

  for (const message of messages) {
    for await (const res of context.prompt(message)) {
      console.log(res)
    }
  }

})();
