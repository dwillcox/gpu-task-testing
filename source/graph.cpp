#include "graph.h"

Graph::Graph(size_t nstates, size_t nhostp, size_t ndevp) {
    graph_finished = false;
    initialize_task_pools(nhostp, ndevp);
    for (Pool* p : device_task_pools) {
        p->tasks.resize(nstates);
        p->checked_out_tasks.resize(nstates);
    }
    task_registry.resize(nstates);
    std::cout << "initialized task pools" << std::endl;
}

Graph::~Graph() {
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

void Graph::set_state_pool_map(std::function<size_t (State*)> map) {
    map_state_to_pool = map;
}

void Graph::queue(State* state) {
    task_registry.push_back(state);
}

bool Graph::completed() {
    bool tasks_unfinished = false;
    for (State* state : task_registry) {
        if (state->status != 3) {
            tasks_unfinished = true;
            break;
        }
    }
    graph_finished = !tasks_unfinished;
    return graph_finished;
}

void Graph::advance(UnifiedVector<State*>& advance_states) {
    std::cout << "in advance ..." << std::endl;

    for (Pool* p : device_task_pools) {
        p->checkin(advance_states, map_state_to_pool);
    }

    for (Pool* p : host_task_pools) {
        p->checkin(advance_states, map_state_to_pool);
    }

    std::cout << "leaving advance ..." << std::endl;
}

void Graph::execute_graph() {
    cudaError_t cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);

    // Initialize task pools with queued tasks in the registry
    std::cout << "initializing task pools..." << std::endl;
    advance(task_registry);

    std::cout << "starting graph execution..." << std::endl;

    while (!completed()) {

        //std::cout << "looping bc not completed" << std::endl;

        //std::cout << "Evaluating if device kernels finished:" << std::endl;
        // check if previous device pool kernels finished and advance states
        int i = 0;
        for (Pool* pool : device_task_pools) {
            if (pool->finished()) {
                //std::cout << "device pool " << i << " is finished, advancing its tasks ..." << std::endl;
                advance(pool->checked_out_tasks);
                pool->reset_checked_out_tasks();
            }
            i++;
        }

        //std::cout << "Evaluating if device kernels ready:" << std::endl;
        // launch device task kernels for pools that are ready
        i = 0;
        for (Pool* pool : device_task_pools) {
            if (pool->ready()) {
                int ntasks = pool->size_queued();
                //std::cout << "got " << ntasks << " ntasks for device pool (ready)" << i << std::endl;
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

        //std::cout << "Evaluating if host kernels finished:" << std::endl;
        // check if previous host pool kernels finished and advance states
        i = 0;
        for (Pool* pool : host_task_pools) {
            if (pool->finished()) {
                //std::cout << "host pool " << i << " is finished, advancing its tasks ..." << std::endl;
                advance(pool->checked_out_tasks);
                pool->reset_checked_out_tasks();
            }
            i++;
        }

        //std::cout << "Evaluating if host kernels ready:" << std::endl;
        // execute host tasks
        i = 0;
        for (Pool* pool : host_task_pools) {
            if (pool->ready()) {
                int ntasks = pool->size_queued();
                //std::cout << "got " << ntasks << " ntasks for host pool (ready)" << i << std::endl;

                // checkout
                pool->checkout();
                pool->set_active();

                // run batched tasks
                State::batched_advance(pool->checked_out_tasks);

                pool->set_inactive();

                i++;
            }
        }

    }

    // sync device
    cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);
}
