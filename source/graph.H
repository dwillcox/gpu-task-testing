#ifndef GRAPH_H
#define GRAPH_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <chrono>
#include "pool.H"
#include "unified.H"
#include "unified_vector.H"

#define VERBOSE_DEBUG 0

template<class MultiStateClass, class StateClass> class Graph : public UnifiedMemoryClass {
public:
    UnifiedVector<Pool<MultiStateClass, StateClass>*> task_pools;
    UnifiedVector<StateClass*> task_registry;
    std::function<int (StateClass*)> map_state_to_pool;
    std::function<bool (StateClass*)> check_state_completed;

    bool graph_started;
    bool graph_finished;
    std::chrono::time_point<std::chrono::high_resolution_clock> graph_start_time, graph_end_time;
  
    Graph(size_t nstates, size_t nhostp, size_t ndevp) {
        graph_started = false;
        graph_finished = false;

        initialize_task_pools(nhostp, ndevp);

        for (Pool<MultiStateClass, StateClass>* p : task_pools) {
            p->tasks.resize(nstates);
        }

        task_registry.resize(nstates);

        std::cout << "initialized task pools" << std::endl;
    }

    ~Graph() {
        for (Pool<MultiStateClass, StateClass>* p : task_pools) {
            delete p;
        }

        for (StateClass* t : task_registry) {
            delete t;
        }
    }

    void start_wallclock() {
        graph_start_time = std::chrono::high_resolution_clock::now();
    }

    void stop_wallclock() {
        graph_end_time = std::chrono::high_resolution_clock::now();
    }

    double get_execution_walltime() {
        if (graph_started && graph_finished)
            return static_cast<double>(std::chrono::duration<double>(graph_end_time-graph_start_time).count());
        else
            return 0.0;
    }

    void initialize_task_pools(size_t num_host_pools, size_t num_device_pools) {
        Pool<MultiStateClass, StateClass>* p;
        int ipool = 0;

        for (int i = 0; i < num_host_pools; i++) {
            p = new Pool<MultiStateClass, StateClass>(ipool);
            p->set_host_pool();
            task_pools.push_back(p);
            ipool++;
        }

        for (int i = 0; i < num_device_pools; i++) {
            p = new Pool<MultiStateClass, StateClass>(ipool);
            p->set_device_pool();
            task_pools.push_back(p);
            ipool++;
        }
    }

    void set_state_pool_map_function(std::function<int (StateClass*)> map) {
        map_state_to_pool = map;
    }

    void set_state_completed_function(std::function<bool (StateClass*)> check) {
        check_state_completed = check;
    }

    void queue(StateClass* state) {
        task_registry.push_back(state);
    }

    bool completed() {
        bool tasks_unfinished = false;
        for (StateClass* state : task_registry) {
            if (!check_state_completed(state)) {
                tasks_unfinished = true;
                break;
            }
        }
        graph_finished = !tasks_unfinished;
        return graph_finished;
    }
    
    void advance(UnifiedVector<StateClass*>& advance_states) {
#if VERBOSE_DEBUG
        std::cout << "in advance ..." << std::endl;
#endif

        for (Pool<MultiStateClass, StateClass>* p : task_pools) {
            p->checkin(advance_states, map_state_to_pool);
        }

#if VERBOSE_DEBUG
        std::cout << "leaving advance ..." << std::endl;
#endif
    }

    void advance(std::vector<MultiStateClass*>& advance_multistates) {
#if VERBOSE_DEBUG
        std::cout << "in advance ..." << std::endl;
#endif

        for (Pool<MultiStateClass, StateClass>* p : task_pools) {
            p->checkin(advance_multistates, map_state_to_pool);
        }

#if VERBOSE_DEBUG
        std::cout << "leaving advance ..." << std::endl;
#endif
    }

    void execute_graph() {
        // Set graph started and initial walltime
        graph_started = true;
        start_wallclock();

        cudaError_t cuda_status = cudaDeviceSynchronize();
        assert(cuda_status == cudaSuccess);

        // Initialize task pools with queued tasks in the registry
        std::cout << "initializing task pools..." << std::endl;
        advance(task_registry);

        // Loop over pools until all tasks are finished
        std::cout << "starting graph execution..." << std::endl;
        while (!completed()) {
            // advance any finished multistates
            for (Pool<MultiStateClass, StateClass>* pool : task_pools) {
                advance(pool->extract_finished());
            }

            // launch multistates from pools for any queued tasks
            for (Pool<MultiStateClass, StateClass>* pool : task_pools) {
                pool->execute_queued();
            }
        }

        // sync device
        cuda_status = cudaDeviceSynchronize();
        assert(cuda_status == cudaSuccess);

        stop_wallclock();
    }
};
#endif
