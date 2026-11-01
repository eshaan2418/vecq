#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace vecq {

inline int default_threads() {
  unsigned hc = std::thread::hardware_concurrency();
  return hc == 0 ? 1 : static_cast<int>(hc);
}

// Morsel-style parallel loop: `threads` workers pull task indices from a
// shared atomic counter, so uneven tasks balance themselves. fn(task, worker)
// receives the worker id for indexing thread-local state. The first exception
// thrown by any task is rethrown on the calling thread.
template <class F>
void parallel_for(size_t tasks, int threads, F&& fn) {
  const int workers = std::max(1, std::min<int>(threads, static_cast<int>(tasks)));
  if (workers <= 1) {
    for (size_t t = 0; t < tasks; ++t) fn(t, 0);
    return;
  }
  std::atomic<size_t> next{0};
  std::exception_ptr err;
  std::mutex err_mu;
  auto body = [&](int w) {
    try {
      for (size_t t = next.fetch_add(1); t < tasks; t = next.fetch_add(1)) fn(t, w);
    } catch (...) {
      std::lock_guard<std::mutex> lk(err_mu);
      if (!err) err = std::current_exception();
      next.store(tasks);
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(workers - 1));
  for (int w = 1; w < workers; ++w) pool.emplace_back(body, w);
  body(0);
  for (auto& t : pool) t.join();
  if (err) std::rethrow_exception(err);
}

}  // namespace vecq
