#include "state.H"

using clock_value_t = long long;

__device__ static void sleep(clock_value_t sleep_cycles) {
    clock_value_t start = clock64();
    clock_value_t elapsed;
    do { elapsed = clock64() - start; }
    while (elapsed < sleep_cycles);
}

__device__
void State::cube() {
    x = x * x * x;
    sleep(1000000000);
}

__host__ __device__
void State::square() {
    x = x * x;
#ifdef __CUDA_ARCH__
    sleep(1000000000);
#endif
}

__device__
void State::advance() {
    if (status == 0) {
        cube();
        counter++;
        if (counter == 3) status = 3;
        else status = 1;
    } else if (status == 1) {
        square();
        counter++;
        if (counter == 3) status = 3;
        else status = 2;      
    } else if (status == 2) {
        square();
        counter++;
        if (counter == 3) status = 3;
        else status = 0;
    }
}

