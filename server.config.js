
const path = require('path');

module.exports = (env, argv) => ({
  output: path.join(__dirname, 'test/dist'),
  client: {
    main: {
      entry: './test/client/index.js',
      uri: '/',
    },
  },
  serverEntry: './test/server/index.ts'
})