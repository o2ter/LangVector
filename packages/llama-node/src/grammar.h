//
//  embedding.h
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

#pragma once

#include "common.h"

class LlamaGrammar : public Napi::ObjectWrap<LlamaGrammar>
{
public:
  grammar_parser::parse_state grammar;

  LlamaGrammar(const Napi::CallbackInfo &info) : Napi::ObjectWrap<LlamaGrammar>(info)
  {
    std::string grammarCode = info[0].As<Napi::String>().Utf8Value();
    grammar = grammar_parser::parse(grammarCode.c_str());
  }

  Napi::Value Description(const Napi::CallbackInfo &info)
  {
    auto file = tmpfile();
    grammar_parser::print_grammar(file, grammar);
    auto size = ftell(file);
    rewind(file);
    std::string desc;
    desc.resize(size);
    fread((char *)desc.c_str(), 1, size, file);
    fclose(file);
    return Napi::String::From(Env(), desc);
  }

  static void init(Napi::Object exports)
  {
    auto def = DefineClass(
        exports.Env(),
        "LlamaGrammar",
        {
            InstanceMethod("description", &LlamaGrammar::Description),
        });
    exports.Set("LlamaGrammar", def);
  }
};

class LlamaGrammarEvaluationState : public Napi::ObjectWrap<LlamaGrammarEvaluationState>
{
public:
  LlamaGrammar *grammar;
  llama_grammar *state;

  LlamaGrammarEvaluationState(const Napi::CallbackInfo &info) : Napi::ObjectWrap<LlamaGrammarEvaluationState>(info)
  {
    grammar = Napi::ObjectWrap<LlamaGrammar>::Unwrap(info[0].As<Napi::Object>());
    grammar->Ref();

    std::vector<const llama_grammar_element *> grammar_rules(grammar->grammar.c_rules());
    state = llama_grammar_init(grammar_rules.data(), grammar_rules.size(), grammar->grammar.symbol_ids.at("root"));
  }

  ~LlamaGrammarEvaluationState()
  {
    grammar->Unref();

    if (state != NULL)
    {
      llama_grammar_free(state);
      state = NULL;
    }
  }

  static void init(Napi::Object exports)
  {
    auto def = DefineClass(
        exports.Env(),
        "LlamaGrammarEvaluationState",
        {
        });
    exports.Set("LlamaGrammarEvaluationState", def);
  }
};
