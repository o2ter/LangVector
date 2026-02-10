//
//  gbnf.ts
//
//  The MIT License
//  Copyright (c) 2021 - 2026 O2ter Limited. All rights reserved.
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

type _Value = boolean | number | string | GBNF;

class GBNF {

  strings: readonly string[];
  values: (_Value | (() => _Value))[];

  constructor(templates: readonly string[], values: (_Value | (() => _Value))[]) {
    this.strings = templates;
    this.values = values;
  }

  toString() {
    const rules: string[] = [];
    const map = new Map<GBNF, string>();
    let counter = 0;
    const parse = (x: _Value) => {
      const value = _.isFunction(x) ? x() : x!;
      if (_.isBoolean(value)) return value ? '"true"' : '"false"';
      if (_.isNumber(value)) return `"${value}"`;
      if (_.isString(value)) return `${JSON.stringify(value)}`;
      if (_.isEqual(value.strings, ['', ''])) return parse(value.values[0]);
      const found = map.get(value);
      if (found) return found;
      const [prefix, ...remain] = value.strings;
      let result = prefix;
      for (const [v, suffix] of _.zip(value.values, remain)) {
        const name = `r${counter++}`;
        rules.push(`${name} ::= ${parse(_.isFunction(v) ? v() : v!)}`);
        map.set(value, name);
        result += name;
        result += suffix;
      }
      return result;
    }
    return [
      `root ::= ${parse(this)}`,
      ...rules,
    ].join('\n');
  }
}

export const gbnf = _.assign((
  templates: TemplateStringsArray,
  ...values: (_Value | (() => _Value))[]
) => new GBNF(templates.raw, values), {
  join: (
    values: _Value[],
    separator: string = ' ',
  ) => new GBNF(['', ...Array(values.length - 1).fill(separator), ''], values),
});
