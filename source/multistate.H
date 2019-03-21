#ifndef MULTISTATE_H
#define MULTISTATE_H
#include "unified_vector.H"
#include "streamcontainer.H"

enum class ExecutionTarget {ExecuteHost, ExecuteDevice};

template<class StateClass> class MultiState : public UnifiedMemoryClass, public StreamContainer, public EventContainer {
    UnifiedVector<StateClass*> states;
    ExecutionTarget target;
    bool execution_finished;
    bool running;
public:
    MultiState(UnifiedVector<StateClass*>& store_states, ExecutionTarget execute_target) {
        states = store_states;
        target = execute_target;
        execution_finished = false;
        running = false;
    }

    ~MultiState() {}

    StateClass** begin() {
        return states.begin();
    }

    StateClass** end() {
        return states.end();
    }

    StateClass*& operator[] (unsigned int i) {
        return states[i];
    }

    __host__ __device__
    size_t size() {
        return states.size();
    }

    bool is_finished() {
        if (!execution_finished && running && target == ExecutionTarget::ExecuteDevice) {
            // check if event has been signaled, if so we're finished
            if (query_cuda_event()) set_finished();
        }
        return execution_finished;
    }

    bool is_running() {
        return running;
    }

    UnifiedVector<StateClass*>& get_states() {
        return states;
    }

    void set_finished() {
        execution_finished = true;
        running = false;
    }

    void set_running() {
        execution_finished = false;
        running = true;
    }

    void execute() {
        set_running();
        advance();
    }

    virtual void advance() {}
};

template<class MultiStateClass> __global__ void multistate_kernel(MultiStateClass* mstate) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  size_t size = mstate->size();
  if (tid < size) {
      (*mstate)[tid]->advance();
  }
}
#endif
