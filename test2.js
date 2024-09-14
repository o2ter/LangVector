const path = require('path');
const { LlamaDevice, Llama3ChatWrapper, defineChatSessionFunction } = require('./dist/index.js');

const modelsDir = path.join(__dirname, './models');

const functions = {
  datetime: defineChatSessionFunction({
    description: "Get current ISO datetime in UTC",
    resultType: { type: 'string' },
    handler() {
      return new Date();
    }
  }),
  randomInt: defineChatSessionFunction({
    description: "Generates a random integer between maximum and minimum inclusively",
    params: {
      type: 'object',
      properties: {
        maximum: { type: 'integer' },
        minimum: { type: 'integer' },
      },
      required: ['maximum', 'minimum'],
    },
    resultType: { type: 'integer' },
    handler({ maximum, minimum }) {
      return Math.floor(Math.random() * (maximum - minimum + 1)) + minimum;
    }
  }),
  randomFloat: defineChatSessionFunction({
    description: "Generates a random floating number between maximum and minimum",
    params: {
      type: 'object',
      properties: {
        maximum: { type: 'number' },
        minimum: { type: 'number' },
      },
      required: ['maximum', 'minimum'],
    },
    resultType: { type: 'number' },
    handler({ maximum, minimum }) {
      return Math.random() * (maximum - minimum) + minimum;
    }
  }),
  todayMenu: defineChatSessionFunction({
    description: "A list of today’s special menu",
    resultType: {
      type: 'object',
      properties: {
        totalCount: { type: 'integer' },
        menus: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              name: { type: 'string' },
              price: { type: 'number' },
            },
            required: ['name', 'price'],
          },
        },
      },
      required: ['totalCount', 'menus'],
    },
    handler() {
      return {
        totalCount: 3,
        menus: [
          {
            name: 'Pizza',
            price: 75,
          },
          {
            name: 'Hamburger',
            price: 80,
          },
          {
            name: 'Fish And Chips',
            price: 75,
          },
        ],
      };
    }
  }),
};

(async () => {

  const model = await LlamaDevice.loadModel({
    modelPath: path.join(modelsDir, 'meta-llama/Meta-Llama-3-8B-Instruct/ggml-model-q3_k_m.gguf'),
  });

  const context = model.createContext({
    contextSize: 4096,
    chatOptions: {
      chatWrapper: new Llama3ChatWrapper,
      functions,
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
