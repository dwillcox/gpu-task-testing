#include <iostream>
#include <cassert>
#include <functional>
#include <random>
#include "graph.H"
#include "state.H"

std::function<int (State*)> create_state_pool_map() {
    // Return value is the global pool index spanning host and device pools.
    // Pools are indexed from 0 to N, with host pools first and then device pools next.
    return [](State* s) -> int {
        if (s->status == step_solve) {return 0;}
        else if (s->status == step_error) {return 1;}
        else if (s->status == step_start || s->status == step_retry) {return 2;}
        else if (s->status == step_update) {return 3;}
        else return -1; // there is no pool -1 so these states won't go into any pool.
    };
}

int main(int argc, char* argv[]) {

    State** state = NULL;
    int size = 10000;

#if EXECUTE_HOST_ONLY
    size_t num_host_pools = 4;
    size_t num_device_pools = 0;    
#else
    size_t num_host_pools = 2;
    size_t num_device_pools = 2;    
#endif

    cudaError_t cuda_status = cudaSuccess;
    cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);  

    cuda_status = cudaMallocManaged(&state, sizeof(State*)*size);
    assert(cuda_status == cudaSuccess);

    Graph* task_graph = new Graph(size, num_host_pools, num_device_pools);

    task_graph->set_state_pool_map_function(create_state_pool_map());

    auto check_state_completed = [](State* s) -> bool {return s->status == integration_finished || s->status == integration_failed;};
    task_graph->set_state_completed_function(check_state_completed);

    cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);

    std::uniform_real_distribution<double> uniform0(0.8, 0.9);
    std::uniform_real_distribution<double> uniform1(0.1, 0.3);
    std::uniform_real_distribution<double> uniform2(0.1, 0.3);
    std::default_random_engine re;
    
    for (int i = 0; i < size; i++) {
        state[i] = new State();
        // state[i]->ynow[0] = 0.8;
        // state[i]->ynow[1] = 0.1;
        // state[i]->ynow[2] = 0.1;

        state[i]->ynow[0] = uniform0(re);
        state[i]->ynow[1] = uniform1(re);
        state[i]->ynow[2] = uniform2(re);

        state[i]->status = step_start;
        task_graph->queue(state[i]);
    }

    task_graph->execute_graph();

    std::cout << "Finished executing graph" << std::endl;
    
    for (int i = 0; i < size; i++) {
        std::cout << "index " << i << std::endl;
        state[i]->print_state();
        delete state[i];
    }

    cuda_status = cudaFree(state);
    assert(cuda_status == cudaSuccess);

    return 0;
}
