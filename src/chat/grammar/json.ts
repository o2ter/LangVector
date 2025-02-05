//
//  json.ts
//
//  The MIT License
//  Copyright (c) 2021 - 2025 O2ter Limited. All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//

import _ from 'lodash';
import { Schema } from '../../context/llama/types/schema';
import { gbnf } from './gbnf';

const DECIMAL_PART = gbnf`[0-9]{1,16}`;
const INTEGRAL_PART = gbnf`[0] | [1-9] [0-9]{0,15}`;

const BOOLEAN = gbnf`"true" | "false"`;
const NUMBER = gbnf`"-"? ${INTEGRAL_PART} ("." ${DECIMAL_PART})? ([eE] [-+]? ${INTEGRAL_PART})?`;
const INTEGER = gbnf`"-"? ${INTEGRAL_PART}`;
const CHAR = gbnf`[^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})`;
const STRING = gbnf`"\\"" ${CHAR}* "\\""`;
const NULL = gbnf`"null"`;

export const schemaToJsonGrammarRules = (schema: Schema, allowedNewline = false) => {

  const SPACE = allowedNewline ? gbnf`| " " | "\\n" [ \\t]*` : gbnf`| " "`;

  const convert = (schema: Schema): ReturnType<typeof gbnf> => {
    if ('type' in schema) {
      switch (schema.type) {
        case 'string': return STRING;
        case 'number': return NUMBER;
        case 'integer': return INTEGER;
        case 'boolean': return BOOLEAN;
        case 'null': return NULL;
        case 'array':
          const value = convert(schema.items);
          return gbnf`"[" ${SPACE} (${value} ${SPACE} ("," ${SPACE} ${value})* ${SPACE})? "]"`;
        case 'object':
          const props = _.mapValues(schema.properties, v => convert(v));
          return gbnf`"{" ${SPACE} ${gbnf.join(
            _.map(
              _.entries(props),
              ([k, v], i) => {
                let s = gbnf`${JSON.stringify(k)} ${SPACE} ":" ${SPACE} ${v} ${SPACE}`;
                if (i !== 0) s = gbnf`"," ${SPACE} ${s}`;
                return _.includes(schema.required, k) ? s : gbnf`${s}?`;
              }
            ),
          )} "}"`
        default: throw Error('Invalid schema');
      }
    }
    if ('const' in schema) {
      if (_.isNil(schema.const)) return NULL;
      return gbnf`${schema.const}`;
    }
    if ('oneOf' in schema) {
      return gbnf.join(_.map(schema.oneOf, x => convert(x)), ' | ');
    }
    throw Error('Invalid schema');
  };
  
  return convert(schema);
}