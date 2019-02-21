#include "graph.H"

#define VERBOSE_DEBUG 0

Graph::Graph(size_t nstates, size_t nhostp, size_t ndevp) {
    multistate = new MultiState(nstates);

    graph_started = false;
    graph_finished = false;
    initialize_task_pools(nhostp, ndevp);
    for (Pool* p : device_task_pools) {
        p->tasks.resize(nstates);
        p->checked_out_tasks.resize(nstates);
    }
    for (Pool* p : host_task_pools) {
        p->tasks.resize(nstates);
        p->checked_out_tasks.resize(nstates);
    }
    task_registry.resize(nstates);
    std::cout << "initialized task pools" << std::endl;
}

Graph::~Graph() {
    delete[] multistate;

    cudaError_t cuda_status = cudaSuccess;
    for (Pool* p : device_task_pools) {
        cuda_status = cudaStreamDestroy(p->get_stream());
        assert(cuda_status == cudaSuccess);
    }
    cuda_status = cudaFree(pool_streams);
    assert(cuda_status == cudaSuccess);

    for (Pool* p : device_task_pools) {
        delete p;
    }

    for (Pool* p : host_task_pools) {
        delete p;
    }

    for (State* t : task_registry) {
        delete t;
    }
}

void Graph::start_wallclock() {
    graph_start_time = std::chrono::high_resolution_clock::now();
}

void Graph::stop_wallclock() {
    graph_end_time = std::chrono::high_resolution_clock::now();
}

double Graph::get_execution_walltime() {
    if (graph_started && graph_finished)
        return static_cast<double>(std::chrono::duration<double>(graph_end_time-graph_start_time).count());
    else
        return 0.0;
}

void Graph::write_statistics() {
    multistate->write_statistics();
}

void Graph::initialize_task_pools(size_t num_host_pools, size_t num_device_pools) {
    Pool* p;
    int ipool = 0;

    for (int i = 0; i < num_host_pools; i++) {
        p = new Pool(ipool);
        p->set_host_pool();
        host_task_pools.push_back(p);
        ipool++;
    }

    for (int i = 0; i < num_device_pools; i++) {
        p = new Pool(ipool);
        p->set_device_pool();
        device_task_pools.push_back(p);
        ipool++;
    }

    // create 1 CUDA stream per task pool and attach it to the pool
    cudaError_t cuda_status = cudaMallocManaged(&pool_streams, device_task_pools.size()*sizeof(cudaStream_t));
    assert(cuda_status == cudaSuccess);
    for (int i = 0; i < device_task_pools.size(); i++) {
        cuda_status = cudaStreamCreate(&(pool_streams[i]));
        assert(cuda_status == cudaSuccess);
        device_task_pools[i]->set_stream(&(pool_streams[i]));
    }
}

void Graph::set_state_pool_map_function(std::function<int (State*)> map) {
    map_state_to_pool = map;
}

void Graph::set_state_completed_function(std::function<bool (State*)> check) {
    check_state_completed = check;
}

void Graph::queue(State* state) {
    task_registry.push_back(state);
}

bool Graph::completed() {
    bool tasks_unfinished = false;
    for (State* state : task_registry) {
        if (!check_state_completed(state)) {
            tasks_unfinished = true;
            break;
        }
    }
    graph_finished = !tasks_unfinished;
    return graph_finished;
}

void Graph::advance(UnifiedVector<State*>& advance_states) {
#if VERBOSE_DEBUG
    std::cout << "in advance ..." << std::endl;
#endif

    for (Pool* p : device_task_pools) {
        p->checkin(advance_states, map_state_to_pool);
    }

    for (Pool* p : host_task_pools) {
        p->checkin(advance_states, map_state_to_pool);
    }

#if VERBOSE_DEBUG
    std::cout << "leaving advance ..." << std::endl;
#endif
}

void Graph::execute_graph() {

    // Set graph started and initial walltime
    graph_started = true;
    start_wallclock();

    cudaError_t cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);

    // Initialize task pools with queued tasks in the registry
    std::cout << "initializing task pools..." << std::endl;
    advance(task_registry);

    std::cout << "starting graph execution..." << std::endl;

    while (!completed()) {

#if VERBOSE_DEBUG
        std::cout << "looping bc not completed" << std::endl;
        std::cout << "Evaluating if device kernels finished:" << std::endl;
#endif

        // check if previous device pool kernels finished and advance states
        int i = 0;
        for (Pool* pool : device_task_pools) {
            if (pool->finished()) {
#if VERBOSE_DEBUG
                std::cout << "device pool " << i << " is finished, advancing its tasks ..." << std::endl;
#endif
                advance(pool->checked_out_tasks);
                pool->reset_checked_out_tasks();
            }
            i++;
        }

#if VERBOSE_DEBUG
        std::cout << "Evaluating if device kernels ready:" << std::endl;
#endif
        // launch device task kernels for pools that are ready
        i = 0;
        for (Pool* pool : device_task_pools) {
            if (pool->ready()) {
                int ntasks = pool->size_queued();
#if VERBOSE_DEBUG
                std::cout << "got " << ntasks << " ntasks for device pool (ready)" << i << std::endl;
#endif
                int numThreads = min(32, ntasks);
                int numBlocks = static_cast<int>(ceil(((double) ntasks)/((double) numThreads)));

                // checkout and copy task pointers to the device
                pool->checkout();
                pool->set_active();

                // launch kernel
                pool_kernel<<<numBlocks, numThreads, 0, pool->get_stream()>>>(pool);
                cudaEventRecord(pool->kernel_finished, pool->get_stream());
            }
            i++;
        }

#if VERBOSE_DEBUG
        std::cout << "Evaluating if host kernels finished:" << std::endl;
#endif
        // check if previous host pool kernels finished and advance states
        i = 0;
        for (Pool* pool : host_task_pools) {
            if (pool->finished()) {
#if VERBOSE_DEBUG
                std::cout << "host pool " << i << " is finished, advancing its tasks ..." << std::endl;
#endif
                advance(pool->checked_out_tasks);
                pool->reset_checked_out_tasks();
            }
            i++;
        }

#if VERBOSE_DEBUG
        std::cout << "Evaluating if host kernels ready:" << std::endl;
#endif
        // execute host tasks
        i = 0;
        for (Pool* pool : host_task_pools) {
            if (pool->ready()) {
                int ntasks = pool->size_queued();
#if VERBOSE_DEBUG
                std::cout << "got " << ntasks << " ntasks for host pool (ready)" << i << std::endl;
#endif

                // checkout
                pool->checkout();
                pool->set_active();

                // run batched tasks
                multistate->advance(pool->checked_out_tasks);

                pool->set_inactive();
            }
            i++;
        }

    }

    // sync device
    cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);

    stop_wallclock();
}
