#ifndef GRAPH_H
#define GRAPH_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "pool.h"
#include "state.h"
#include "unified.h"
#include "unified_vector.h"


class Graph : public UnifiedMemoryClass {
public:
    UnifiedVector<Pool*> device_task_pools;
    UnifiedVector<Pool*> host_task_pools;  
    UnifiedVector<State*> task_registry;
    std::function<int (State*)> map_state_to_pool;
    std::function<bool (State*)> check_state_completed;
    cudaStream_t* pool_streams;

    bool graph_finished;
  
    Graph(size_t, size_t, size_t);

    ~Graph();

    void initialize_task_pools(size_t, size_t);

    void set_state_pool_map_function(std::function<int (State*)>);

    void set_state_completed_function(std::function<bool (State*)>);

    void queue(State*);

    bool completed();
    
    void advance(UnifiedVector<State*>&);

    void execute_graph();
};
#endif
