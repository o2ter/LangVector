import _ from 'lodash';
import resolve from '@rollup/plugin-node-resolve';
import typescript from '@rollup/plugin-typescript';
import commonjs from '@rollup/plugin-commonjs';
import babel from '@rollup/plugin-babel';
import json from '@rollup/plugin-json';
import sass from 'rollup-plugin-sass';
import dts from 'rollup-plugin-dts';

const rollupPlugins = [
  typescript({ declaration: false }),
  babel({
    babelrc: false,
    exclude: 'node_modules/**',
    babelHelpers: 'bundled',
  }),
  commonjs({
    transformMixedEsModules: true,
  }),
  json(),
  sass({
    output: true,
  }),
];

const rollupConfig = {
  input: 'src/index',
  external: [
    /node_modules/,
    /^react$/,
    /^react-native$/,
  ],
};

export default [
  {
    ...rollupConfig,
    output: [
      {
        file: 'dist/index.js',
        format: 'cjs',
        sourcemap: true,
      },
      {
        file: 'dist/index.mjs',
        format: 'esm',
        sourcemap: true,
      },
    ],
    plugins: [
      resolve({
        extensions: [
          '.tsx', '.jsx',
          '.ts', '.mjs', '.js',
        ]
      }),
      ...rollupPlugins
    ],
  },
  {
    ...rollupConfig,
    output: [
      {
        file: `dist/index.d.ts`,
        format: 'es',
      },
    ],
    plugins: [
      resolve({
        extensions: [
          '.tsx', '.jsx',
          '.ts', '.mjs', '.js',
        ]
      }),
      dts()
    ],
  }
];