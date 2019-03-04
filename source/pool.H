#ifndef POOL_H
#define POOL_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <functional>
#include <mutex>
#include "unified_vector.H"
#include "state.H"
#include "unified.H"
#include "streamcontainer.H"

// As a first pass, let's use Pool-wide locking
class Pool : public UnifiedMemoryClass, public StreamContainer {
public:
  std::mutex pool_mutex;
  UnifiedVector<State*> tasks;
  UnifiedVector<State*> checked_out_tasks;
  int pool_graph_index;
  bool _is_device_pool;
  bool kernel_running;
  cudaEvent_t kernel_finished;

  Pool(int index) {
    pool_graph_index = index;
    _is_device_pool = false;
    kernel_running = false;
    cudaEventCreate(&kernel_finished);
  }

  ~Pool() {
  }

  void set_host_pool() {
    _is_device_pool = false;
  }

  void set_device_pool() {
    _is_device_pool = true;
  }

  bool is_host_pool() {
    return !_is_device_pool;
  }

  bool is_device_pool() {
    return _is_device_pool;
  }

  bool ready() {
    return !kernel_running && (size_queued() > 0);
  }

    bool finished() {
        if (is_device_pool()) {
            if (kernel_running) {
                cudaError_t cuda_status = cudaEventQuery(kernel_finished);
                if (cuda_status == cudaSuccess) {
                    set_inactive();
                }
            }
        }
        return !kernel_running && (size_checked_out() > 0);
    }

  void set_active() {
    kernel_running = true;
  }

  void set_inactive() {
    kernel_running = false;
  }

  size_t size_queued() {
    return tasks.size();
  }

  size_t size_checked_out() {
    return checked_out_tasks.size();
  }

  void checkin(UnifiedVector<State*>& checkin_states, std::function<int (State*)> pool_map) {
    std::unique_lock<std::mutex> pool_lock(pool_mutex);
    for (State* s : checkin_states) {
        if (pool_map(s) == pool_graph_index) {
            tasks.push_back(s);
        }
    }
  }

  void checkout() {
    std::unique_lock<std::mutex> pool_lock(pool_mutex);
    assert(checked_out_tasks.size() == 0);
    checked_out_tasks = tasks;
    tasks.resize(0);
  }

  void reset_tasks() {
    std::unique_lock<std::mutex> pool_lock(pool_mutex);
    tasks.resize(0);
  }

  void reset_checked_out_tasks() {
    std::unique_lock<std::mutex> pool_lock(pool_mutex);
    checked_out_tasks.resize(0);
  }
};

__global__ void pool_kernel(Pool*);
#endif
