
const path = require('path');
const webpack = require('webpack');

module.exports = (env, argv) => ({
  output: path.join(__dirname, 'test/dist'),
  client: {
    main: {
      entry: './test/client/index.js',
      uri: '/',
    },
  },
  serverEntry: './test/server/index.ts',
  serverAssets: [
    { from: 'packages/llama-node/build/Release/!(*.node)', to: '[name][ext]' },
  ],
  options: {
    plugins: [
      new webpack.IgnorePlugin({ resourceRegExp: /^pg-native$/ }),
    ],
  },
})