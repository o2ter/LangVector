
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
  options: {
    plugins: [
      new webpack.IgnorePlugin({ resourceRegExp: /^pg-native$/ }),
    ],
    externals: {
      '@reflink/reflink': 'commonjs2 @reflink/reflink',
      '@node-llama-cpp/win-x64-cuda': 'commonjs2 @node-llama-cpp/win-x64-cuda',
      '@node-llama-cpp/linux-x64-cuda': 'commonjs2 @node-llama-cpp/linux-x64-cuda',
    },
  },
})