const path = require('path');
const { LlamaDevice, Llama3ChatWrapper } = require('./dist/index.js');

const modelsDir = path.join(__dirname, './models');

(async () => {

  const model = await LlamaDevice.loadModel({
    modelPath: path.join(modelsDir, 'meta-llama/Meta-Llama-3-8B-Instruct/ggml-model-q3_k_m.gguf'),
  });

  const context = model.createContext({
    contextSize: 4096,
    chatOptions: {
      chatWrapper: new Llama3ChatWrapper,
    },
  });

  const messages = [
    'hi',
  ];

  for (const message of messages) {
    for await (const { response, done } of context.prompt(message)) {
      console.log({ response: model.detokenize(response, { decodeSpecial: true }), done })
    }
  }

})();
