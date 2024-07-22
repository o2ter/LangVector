//
//  typescript.ts
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
import { ChatModelFunctionOptions } from '../../context/llama/types';

export const _typeScriptSchemaString = (schema: Schema): string => {
  if ('type' in schema) {
    switch (schema.type) {
      case 'string': return 'string';
      case 'number': return 'number';
      case 'integer': return 'bigint';
      case 'boolean': return 'boolean';
      case 'null': return 'null';
      case 'array': return `${_typeScriptSchemaString(schema.items)}[]`;
      case 'object':
        const props: string[] = [];
        for (const [key, value] of _.entries(schema.properties)) {
          if (_.includes(schema.required, key)) {
            props.push(`${key}: ${_typeScriptSchemaString(value)};`);
          } else {
            props.push(`${key}?: ${_typeScriptSchemaString(value)};`);
          }
        }
        return `{ ${props.join(' ')} }`;
      default: throw Error('Invalid schema');
    }
  }
  if ('const' in schema) {
    return JSON.stringify(schema.const);
  }
  if ('oneOf' in schema) {
    return _.map(schema.oneOf, x => _typeScriptSchemaString(x)).join(' | ');
  }
  throw Error('Invalid schema');
}

export const _typeScriptFunctionSignatures = (functions: Record<string, ChatModelFunctionOptions>): string => {
  const result: string[] = [];
  for (const [name, options] of _.entries(functions)) {
    const desc = options.description?.split(/\r\n|\r|\n/).map(x => `// ${x.trim()}`) ?? [];
    const func = options.params ? `function ${name}(params: ${_typeScriptSchemaString(options.params)})` : `function ${name}()`;
    const res = options.resultType ? `: ${_typeScriptSchemaString(options.resultType)}` : '';
    result.push([
      ...desc,
      `${func}${res}`,
    ].join('\n'));
  }
  return result.join('\n\n');
}
