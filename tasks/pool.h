#ifndef POOL_H
#define POOL_H
#include "generic_vector.h"
#include "state.h"
#include "lock.h"
#include "unified.h"
#include "streamcontainer.h"
#include "graph.h"

// As a first pass, let's use Pool-wide locking
class Pool : public UnifiedMemoryClass, public StreamContainer {
  HostDeviceLock* lock;
public:
  GenericVector<State*> tasks;  
  int pool_graph_index;
  bool supports_host_execution;
  bool supports_device_execution;
  
  Pool(int index) {
    lock = new HostDeviceLock();
    pool_graph_index = index;
    supports_host_execution = false;
    supports_device_execution = false;
  }

  ~Pool() {
    delete lock;
  }

  void checkin(State* state) {
    lock->lock();
    tasks.push_back(state);
    lock->unlock();
  }

  void reset() {
    lock->lock();
    tasks.resize(0);
    lock->unlock();
  }
};
#endif
