#ifndef POOL_H
#define POOL_H
#include <cuda.h>
#include <cuda_runtime.h>
#include "unified_vector.h"
#include "state.h"
#include "lock.h"
#include "unified.h"
#include "streamcontainer.h"
#include "graph.h"

// As a first pass, let's use Pool-wide locking
class Pool : public UnifiedMemoryClass, public StreamContainer {
  HostDeviceLock* lock;
public:
  UnifiedVector<State*> tasks;
  UnifiedVector<State*> checked_out_tasks;
  int pool_graph_index;
  bool _is_device_pool;
  bool kernel_running;
  cudaEvent_t kernel_finished;  
  
  Pool(int index) {
    lock = new HostDeviceLock();
    pool_graph_index = index;
    _is_device_pool = false;
    kernel_running = false;
    cudaEventCreate(&kernel_finished);
  }

  ~Pool() {
    delete lock;
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

  void checkin(UnifiedVector<State*>* checkin_states, std::function<bool(State*)> test) {
    lock->lock();
    std::cout << "about to loop" << std::endl;
    std::cout << "filled size is " << checkin_states->filled_size << std::endl;
    std::cout << "allocated size is " << checkin_states->allocated_size << std::endl;    
    //    for (State* s : checkin_states) {
    for (int i = 0; i < checkin_states->filled_size; i++) {
        State* s = (*checkin_states)[i];
        std::cout << "testing if push back with status " << s->status << std::endl;
        if (test(s)) {
            std::cout << "calling push back" << std::endl;
            tasks.push_back(s);
        }
    }
    lock->unlock();
  }

  void checkout() {
    lock->lock();
    assert(checked_out_tasks.size() == 0);
    checked_out_tasks = tasks;
    tasks.resize(0);
    lock->unlock();
  }

  void reset_tasks() {
    lock->lock();
    tasks.resize(0);
    lock->unlock();
  }

  void reset_checked_out_tasks() {
    lock->lock();
    checked_out_tasks.resize(0);
    lock->unlock();
  }  
};
#endif
