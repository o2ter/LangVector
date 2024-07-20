//
//  json.ts
//
//  The MIT License
//  Copyright (c) 2021 - 2024 O2ter Limited. All rights reserved.
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

// object: new GrammarRule('"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space', ['string', 'value']),
//   array: new GrammarRule('"[" space ( value ("," space value)* )? "]" space', ['value']),

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
          
        case 'object':
          
        default: throw Error('Invalid schema');
      }
    }
    if ('const' in schema) {
      if (_.isNil(schema.const)) return NULL;
      return gbnf`${schema.const}`;
    }
    if ('oneOf' in schema) {
      return gbnf.oneOf(..._.map(schema.oneOf, x => convert(x)));
    }
    throw Error('Invalid schema');
  };
  
  return convert(schema);
}