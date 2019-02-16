#include "state.H"

const double State::atol = 1.e-12;

void State::initialize_csr_jac() {
    csr_jac_col_index[0] = 1;
    csr_jac_col_index[1] = 2;
    csr_jac_col_index[2] = 3;
    csr_jac_col_index[3] = 1;
    csr_jac_col_index[4] = 2;
    csr_jac_col_index[5] = 3;
    csr_jac_col_index[6] = 2;
    csr_jac_col_index[7] = 3;

    csr_jac_row_count[0] = 1;
    csr_jac_row_count[1] = 4;
    csr_jac_row_count[2] = 7;
    csr_jac_row_count[3] = 9;
}

__host__ __device__
void State::rhs() {
    // Evaluates rhs at ynext
    dydt[0] = -.04e0*ynext[0] + 1.e4*ynext[1]*ynext[2];
    dydt[2] = 3.e7*ynext[1]*ynext[1];
    dydt[1] = -dydt[0]-dydt[2];
    #if (VERBOSE_DEBUG && !defined(__CUDA_ARCH__))
    std::cout << "in State::rhs" << std::endl;
    print_state();
    #endif
}

__host__ __device__
void State::jac() {
    // Evaluates jacobian at ynext
    dfdy[0] = -0.04e0;
    dfdy[1] = 1.e4*ynext[2];
    dfdy[2] = 1.e4*ynext[1];
    dfdy[6] = 6.0e7*ynext[1];
    dfdy[3] = 0.04e0;
    dfdy[4] = -dfdy[1]-dfdy[6];
    dfdy[5] = -dfdy[2];
    dfdy[7] = 0.0e0;
    #if (VERBOSE_DEBUG && !defined(__CUDA_ARCH__))
    std::cout << "in State::jac" << std::endl;
    print_state();
    #endif    
}

__host__ __device__
void State::jac_scale_add_identity() {
    #if (VERBOSE_DEBUG && !defined(__CUDA_ARCH__))
    std::cout << std::scientific << "in jac_scale with dt = " << dt << std::endl;
    #endif
    // first scale by -dt
    for (size_t i = 0; i < sparse_jac_nnz; i++) {
        dfdy[i] = dfdy[i] * (-dt);
    }
    
    // the base-1 indexes here are only logical
    size_t location = 0;
    size_t num_in_row;
    size_t col_index;
    for (size_t row_index = 1; row_index <= neqs; row_index++) {
        num_in_row = csr_jac_row_count[row_index] - csr_jac_row_count[row_index-1];
        for (size_t counter = 0; counter < num_in_row; counter++) {
            col_index = csr_jac_col_index[location];
            if (row_index == col_index)
                dfdy[location] += 1.0;
            location++;
        }
    }
}

__host__ __device__
void State::set_matrix_rhs() {
    for (size_t i = 0; i < neqs; i++) bmat[i] = ynow[i] + dt * dydt[i] - ynext[i];
}

__host__ __device__
void State::setup_step() {
    if (status == step_start) {
        num_convergence_iters = 0;
        for (size_t i = 0; i < neqs; i++) ynext[i] = ynow[i];
    }
}

__host__ __device__
void State::setup_matrix() {
    jac();
    jac_scale_add_identity();
    rhs();
    set_matrix_rhs();
}

__host__ __device__
void State::post_step() {
    // Check if converged
    bool converged = true;
    for (size_t i = 0; i < neqs; i++) {
        converged = dynext[i] < atol;
        if (!converged) break;
    }

    if (converged) {
        // Determine next action if the current step converged        
        if (time >= end_time)
            status = integration_finished;
        else {
            for (size_t i = 0; i < neqs; i++) ynow[i] = ynext[i] + dynext[i];
            dt = min(dt, end_time - time);
            time += dt;
            status = step_start;
        }
    } else {
        // Determine next action if the current step failed to converge
        if (num_convergence_iters == max_convergence_iters) {
            status = step_error;
        } else {
            for (size_t i = 0; i < neqs; i++) ynext[i] += dynext[i];
            num_convergence_iters++;
            status = step_retry;
        }
    }
}

__host__ __device__
void State::advance() {
#if (VERBOSE_DEBUG && !defined(__CUDA_ARCH__))
    if (status == step_start)
        std::cout << "... starting a new step" << std::endl;
    else if (status == step_retry)
        std::cout << "... retrying previous step" << std::endl;
#endif
    if (status == step_start || status == step_retry) {
        setup_step();
        setup_matrix();
        status = step_solve;
    } else if (status == step_update) {
        post_step();
    }
}

void State::print_state() {
    std::cout << std::scientific;
    std::cout << "State with status: " << status << std::endl;
    std::cout << "   - ynow = ";
    for (size_t i = 0; i < neqs; i++) std::cout << ynow[i] << " ";
    std::cout << "   - ynext = ";
    for (size_t i = 0; i < neqs; i++) std::cout << ynext[i] << " ";
    std::cout << "   - dynext = ";
    for (size_t i = 0; i < neqs; i++) std::cout << dynext[i] << " ";
    std::cout << "   - dydt = ";
    for (size_t i = 0; i < neqs; i++) std::cout << dydt[i] << " ";
    std::cout << "   - dfdy = ";
    for (size_t i = 0; i < State::sparse_jac_nnz; i++) std::cout << dfdy[i] << " ";
    std::cout << "   - bmat = ";
    for (size_t i = 0; i < neqs; i++) std::cout << bmat[i] << " ";    
    std::cout << std::endl << std::endl;
}

void State::report_error() {
    std::cout << "Integration Failed for state with:" << std::endl;
    std::cout << "   - yn = ";
    for (size_t i = 0; i < neqs; i++) std::cout << ynow[i] << " ";
    std::cout << std::endl << std::endl;
    status = integration_failed;
}
