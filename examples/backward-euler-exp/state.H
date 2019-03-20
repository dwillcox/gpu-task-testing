#ifndef STATE_H
#define STATE_H
#include <cassert>
#include "unified_vector.H"

enum IntegratorStatus {step_start,
                       step_solve,
                       step_update,
                       step_retry,
                       integration_failed,
                       integration_finished};

class State : public UnifiedMemoryClass {
public:
    static const size_t neqs = 3;
    static const size_t max_convergence_iters = 10;
    static const double atol;

    // Stopping Criteria
    double time, dt, end_time;
    size_t num_convergence_iters;

    // Solutions and iteration delta
    double ynow[State::neqs];
    double ynext[State::neqs];
    double dynext[State::neqs];

    // RHS
    double dydt[State::neqs];

    // Jacobian & Matrix system
    static const int sparse_jac_nnz = 3;
    double dfdy[State::sparse_jac_nnz];
    double bmat[State::neqs];

    int csr_jac_col_index[State::sparse_jac_nnz];
    int csr_jac_row_count[State::neqs+1];

    // State status
    IntegratorStatus status;

    State() {
        std::cout << "state constructor" << std::endl;
        status = step_start;
        time = 0.0;
        dt = 1.0e-1;
        end_time = 1.0;
        initialize_csr_jac();
    }

    void initialize_csr_jac();

    __host__ __device__
    void rhs();

    __host__ __device__
    void jac();

    __host__ __device__
    void jac_scale_add_identity();

    __host__ __device__
    void set_matrix_rhs();    

    __host__ __device__
    void setup_step();

    __host__ __device__
    void setup_matrix();

    __host__ __device__
    void post_step();

    __host__ __device__
    void advance();

    void print_state();

    void report_error();
};
#endif
