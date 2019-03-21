#ifndef POOL_H
#define POOL_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <functional>
#include <mutex>
#include <vector>
#include "unified_vector.H"
#include "unified.H"

template<class MultiStateClass, class StateClass> class Pool : public UnifiedMemoryClass {
public:
  std::mutex pool_mutex;
  UnifiedVector<StateClass*> tasks;
  std::vector<MultiStateClass*> multistates;

  int pool_graph_index;
  bool _is_device_pool;

  Pool(int index) {
    pool_graph_index = index;
    _is_device_pool = false;
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
      return (size_queued() > 0);
  }

  size_t size_queued() {
    return tasks.size();
  }

  size_t size_checked_out() {
    return multistates.size();
  }

    std::vector<MultiStateClass*> extract_finished() {
        std::vector<MultiStateClass*> unfinished_multistates;
        std::vector<MultiStateClass*> finished_multistates;

        for (auto& ms : multistates) {
            if (ms->is_finished())
                finished_multistates.push_back(ms);
            else
                unfinished_multistates.push_back(ms);
        }

        multistates = unfinished_multistates;
        return finished_multistates;
    }

  void checkin(UnifiedVector<StateClass*>& checkin_states, std::function<int (StateClass*)> pool_map) {
    std::unique_lock<std::mutex> pool_lock(pool_mutex);
    for (StateClass* s : checkin_states) {
        if (pool_map(s) == pool_graph_index) {
            tasks.push_back(s);
        }
    }
  }

    void checkin(std::vector<MultiStateClass*>& checkin_multistates, std::function<int (StateClass*)> pool_map) {
        std::unique_lock<std::mutex> pool_lock(pool_mutex);
        for (MultiStateClass* ms : checkin_multistates) {
            for (StateClass* s : *ms) {
                if (pool_map(s) == pool_graph_index) {
                    tasks.push_back(s);
                }
            }
        }
    }

    void execute_queued() {
        std::unique_lock<std::mutex> pool_lock(pool_mutex);

        if (ready()) {
            ExecutionTarget target = is_device_pool() ? ExecutionTarget::ExecuteDevice : ExecutionTarget::ExecuteHost;

            MultiStateClass* ms = new MultiStateClass(tasks, target);
            multistates.push_back(ms);

            tasks.resize(0);

            ms->execute();
        }
    }

  void reset_tasks() {
    std::unique_lock<std::mutex> pool_lock(pool_mutex);
    tasks.resize(0);
  }
};
#endif
