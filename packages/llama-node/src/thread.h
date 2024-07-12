//
//  thread.h
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

#include <queue>

class ThreadPool
{
public:
  static ThreadPool &shared()
  {
    static ThreadPool instance;
    return instance;
  }

  static void excute(const std::function<void()> &job)
  {
    shared()._excute(job);
  }

  void release()
  {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      should_terminate = true;
    }
    mutex_condition.notify_all();
    for (std::thread &active_thread : threads)
    {
      active_thread.join();
    }
    threads.clear();
  }

private:
  bool should_terminate = false;
  std::mutex queue_mutex;
  std::condition_variable mutex_condition;
  std::vector<std::thread> threads;
  std::queue<std::function<void()>> jobs;

  ThreadPool()
  {
    const uint32_t num_threads = std::thread::hardware_concurrency();
    for (uint32_t ii = 0; ii < num_threads; ++ii)
    {
      threads.emplace_back(std::thread(&ThreadPool::loop, this));
    }
  }

  void loop()
  {
    while (true)
    {
      std::function<void()> job;
      {
        std::unique_lock<std::mutex> lock(queue_mutex);
        mutex_condition.wait(lock, [this]
                             { return !jobs.empty() || should_terminate; });
        if (should_terminate)
        {
          return;
        }
        job = jobs.front();
        jobs.pop();
      }
      job();
    }
  }

  void _excute(const std::function<void()> &job)
  {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      jobs.push(job);
    }
    mutex_condition.notify_one();
  }
};

Napi::TypedThreadSafeFunction<Napi::Reference<Napi::Function>> ThreadSafeCallback(
    Napi::Env env,
    const std::function<void(const Napi::CallbackInfo &info)> &callback)
{
  return Napi::TypedThreadSafeFunction<Napi::Reference<Napi::Function>>::New(
      env,
      Napi::Function::New(env, callback),
      "ThreadSafeCallback",
      0,
      1);
}
