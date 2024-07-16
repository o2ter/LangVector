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
#include "context.h"

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
  LlamaContext *ctx;
  LlamaGrammar *grammar;
  llama_grammar *state;

  LlamaGrammarEvaluationState(const Napi::CallbackInfo &info) : Napi::ObjectWrap<LlamaGrammarEvaluationState>(info)
  {
    ctx = Napi::ObjectWrap<LlamaContext>::Unwrap(info[0].As<Napi::Object>());
    ctx->Ref();
    grammar = Napi::ObjectWrap<LlamaGrammar>::Unwrap(info[1].As<Napi::Object>());
    grammar->Ref();

    std::vector<const llama_grammar_element *> grammar_rules(grammar->grammar.c_rules());
    state = llama_grammar_init(grammar_rules.data(), grammar_rules.size(), grammar->grammar.symbol_ids.at("root"));
  }

  ~LlamaGrammarEvaluationState()
  {
    ctx->Unref();
    grammar->Unref();

    if (state != NULL)
    {
      llama_grammar_free(state);
      state = NULL;
    }
  }

  Napi::Value SampleToken(const Napi::CallbackInfo &info)
  {
    auto candidates = Napi::ObjectWrap<LlamaContextSampleCandidates>::Unwrap(info[0].As<Napi::Object>());
    llama_token_data_array candidates_p = {candidates->candidates.data(), candidates->candidates.size(), false};
    llama_sample_grammar(ctx->ctx, &candidates_p, state);
    if (candidates_p.size < candidates->candidates.size())
    {
      candidates->candidates.resize(candidates_p.size);
    }
  }

  Napi::Value AcceptToken(const Napi::CallbackInfo &info)
  {
    llama_token tokenId = info[0].As<Napi::Number>().Int32Value();
    llama_grammar_accept_token(ctx->ctx, state, tokenId);
    return info.Env().Undefined();
  }

  Napi::Value CanBeNextToken(const Napi::CallbackInfo &info)
  {
    llama_token tokenId = info[0].As<Napi::Number>().Int32Value();

    std::vector<llama_token_data> candidates;
    candidates.reserve(1);
    candidates.emplace_back(llama_token_data{tokenId, 1, 0.0f});

    llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

    llama_sample_grammar(ctx->ctx, &candidates_p, state);

    if (candidates_p.size == 0 || candidates_p.data[0].logit == -INFINITY)
    {
      return Napi::Boolean::New(info.Env(), false);
    }

    return Napi::Boolean::New(info.Env(), true);
  }

  static void init(Napi::Object exports)
  {
    auto def = DefineClass(
        exports.Env(),
        "LlamaGrammarEvaluationState",
        {
            InstanceMethod("sampleToken", &LlamaGrammarEvaluationState::SampleToken),
            InstanceMethod("acceptToken", &LlamaGrammarEvaluationState::AcceptToken),
            InstanceMethod("canBeNextToken", &LlamaGrammarEvaluationState::CanBeNextToken),
        });
    exports.Set("LlamaGrammarEvaluationState", def);
  }
};
