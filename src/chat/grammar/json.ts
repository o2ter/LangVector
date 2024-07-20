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
import { GrammarRule, GrammarRuleSet } from './utils';

const SPACE_RULE = new GrammarRule('| " "');
const SPACE_AND_NEWLINE_RULE = new GrammarRule('| " " | "\\n" [ \\t]{0,20}');

const PRIMITIVE_RULES = {
  boolean: new GrammarRule('("true" | "false") space'),
  'decimal-part': new GrammarRule('[0-9]{1,16}'),
  'integral-part': new GrammarRule('[0] | [1-9] [0-9]{0,15}'),
  number: new GrammarRule('("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space', ['integral-part', 'decimal-part']),
  integer: new GrammarRule('("-"? integral-part) space', ['integral-part']),
  value: new GrammarRule('object | array | string | number | boolean | null', ['object', 'array', 'string', 'number', 'boolean', 'null']),
  object: new GrammarRule('"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space', ['string', 'value']),
  array: new GrammarRule('"[" space ( value ("," space value)* )? "]" space', ['value']),
  char: new GrammarRule(`[^"\\\\\\x7F\\x00-\\x1F] | [\\\\] (["\\\\bfnrt] | "u" [0-9a-fA-F]{4})`),
  string: new GrammarRule(`"\\"" char* "\\"" space`, ['char']),
  null: new GrammarRule('"null" space'),
};

const STRING_FORMAT_RULES = {
  'date': new GrammarRule('[0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( \"0\" [1-9] | [1-2] [0-9] | "3" [0-1] )'),
  'time': new GrammarRule('([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )'),
  'date-time': new GrammarRule('date "T" time', ['date', 'time']),
  'date-string': new GrammarRule('"\\"" date "\\"" space', ['date']),
  'time-string': new GrammarRule('"\\"" time "\\"" space', ['time']),
  'date-time-string': new GrammarRule('"\\"" date-time "\\"" space', ['date-time']),
  'uuid': new GrammarRule('[0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12}'),
  'uuid-string': new GrammarRule('"\\"" uuid "\\"" space', ['uuid']),
};

export const schemaToJsonGrammarRules = (schema: Schema, allowedNewline = false): GrammarRuleSet => {

  const space = allowedNewline ? SPACE_AND_NEWLINE_RULE : SPACE_RULE;


  return {
    space,
  };
}