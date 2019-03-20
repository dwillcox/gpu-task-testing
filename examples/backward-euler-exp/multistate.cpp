#include "multistate.H"

MultiState::MultiState(size_t max_states) {
    std::cout << "constructing multistate!" << std::endl;
    
    // initialize batched matrix memory
    cudaError_t cuda_status = cudaSuccess;

    cuda_status = cudaMalloc((void**) &matrices_csr_values,
                             sizeof(double) * State::sparse_jac_nnz * max_states);
    assert(cuda_status == cudaSuccess);

    cuda_status = cudaMalloc((void**) &system_x,
                             sizeof(double) * State::neqs * max_states);
    assert(cuda_status == cudaSuccess);

    cuda_status = cudaMalloc((void**) &system_b,
                             sizeof(double) * State::neqs * max_states);
    assert(cuda_status == cudaSuccess);

    // initialize the CSR matrix data
    matrices_csr_col_index_h[0] = 1;
    matrices_csr_col_index_h[1] = 2;
    matrices_csr_col_index_h[2] = 3;

    matrices_csr_row_count_h[0] = 1;
    matrices_csr_row_count_h[1] = 2;
    matrices_csr_row_count_h[2] = 3;
    matrices_csr_row_count_h[3] = 4;

    cuda_status = cudaMalloc((void**) &matrices_csr_col_index_d,
                             sizeof(int) * State::sparse_jac_nnz);
    assert(cuda_status == cudaSuccess);

    cuda_status = cudaMalloc((void**) &matrices_csr_row_count_d,
                             sizeof(int) * (State::neqs+1));
    assert(cuda_status == cudaSuccess);

    cuda_status = cudaMemcpy(matrices_csr_col_index_d, matrices_csr_col_index_h,
                             sizeof(int) * State::sparse_jac_nnz,
                             cudaMemcpyHostToDevice);
    assert(cuda_status == cudaSuccess);

    cuda_status = cudaMemcpy(matrices_csr_row_count_d, matrices_csr_row_count_h,
                             sizeof(int) * (State::neqs+1),
                             cudaMemcpyHostToDevice);
    assert(cuda_status == cudaSuccess);

    // initialize the linear solver memory
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cusparseStatus_t cusparse_status = CUSPARSE_STATUS_SUCCESS;

    // Make handle for cuSolver
#if PRINT_CUSOLVER_DEBUGGING
    std::cout << "Creating cuSolver Handle" << std::endl;
#endif

    cusolver_status = cusolverSpCreate(&cusolverHandle);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    // Create an info object
#if PRINT_CUSOLVER_DEBUGGING
    std::cout << "Creating info object" << std::endl;
#endif

    cusolver_status = cusolverSpCreateCsrqrInfo(&info);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    // Setup Sparse system description
#if PRINT_CUSOLVER_DEBUGGING
    std::cout << "Creating Matrix Descriptor" << std::endl;
#endif

    cusparse_status = cusparseCreateMatDescr(&system_description);
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

#if PRINT_CUSOLVER_DEBUGGING
    std::cout << "In Matrix Descriptor, setting Matrix Type" << std::endl;
#endif
    cusparse_status = cusparseSetMatType(system_description, CUSPARSE_MATRIX_TYPE_GENERAL);
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

#if PRINT_CUSOLVER_DEBUGGING
    std::cout << "In Matrix Descriptor, setting Matrix Index Base" << std::endl;
#endif
    cusparse_status = cusparseSetMatIndexBase(system_description, CUSPARSE_INDEX_BASE_ONE);
    assert(cusparse_status == CUSPARSE_STATUS_SUCCESS);

#if PRINT_CUSOLVER_DEBUGGING
    std::cout << "Created CV_cuSolver_Mem object." << std::endl;
#endif
    
    // analyze matrix system
    cusolver_status = CUSOLVER_STATUS_SUCCESS;

    std::cout << "max states: " << max_states << std::endl;

    cusolver_status = cusolverSpXcsrqrAnalysisBatched(cusolverHandle,
                                                      State::neqs, // size per subsystem
                                                      State::neqs, // size per subsystem
                                                      State::sparse_jac_nnz,
                                                      system_description,
                                                      matrices_csr_row_count_d,
                                                      matrices_csr_col_index_d,
                                                      info);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cusolver_status = cusolverSpDcsrqrBufferInfoBatched(cusolverHandle,
                                                        State::neqs, // size per subsystem
                                                        State::neqs, // size per subsystem
                                                        State::sparse_jac_nnz,
                                                        system_description,
                                                        matrices_csr_values,
                                                        matrices_csr_row_count_d,
                                                        matrices_csr_col_index_d,
                                                        max_states,
                                                        info,
                                                        &internal_size,
                                                        &workspace_size);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cuda_status = cudaMalloc((void**) &workspace, workspace_size);
    assert(cuda_status == cudaSuccess);
}

MultiState::~MultiState() {
    // free batched matrix memory
    cudaError_t cuda_status = cudaSuccess;

    cuda_status = cudaFree(matrices_csr_values);
    assert(cuda_status == cudaSuccess);

    cuda_status = cudaFree(system_x);
    assert(cuda_status == cudaSuccess);

    cuda_status = cudaFree(system_b);
    assert(cuda_status == cudaSuccess);

    // free CSR matrix data
    cuda_status = cudaFree(matrices_csr_col_index_d);
    assert(cuda_status == cudaSuccess);

    cuda_status = cudaFree(matrices_csr_row_count_d);
    assert(cuda_status == cudaSuccess);

    // free the linear solver memory
    cuda_status = cudaFree(workspace);
    assert(cuda_status == cudaSuccess);
}

void MultiState::states_to_solver(UnifiedVector<State*>& batched_states) {
    cudaError_t cuda_status = cudaSuccess;

    // load the batched state matrix systems into the solver memory
    int i = 0;
    for (State* s : batched_states) {
        // copy matrix A
        std::cout << "reporting before solver: " << i << std::endl;        
        s->print_state();
        double* matrix_destination = matrices_csr_values + i * State::sparse_jac_nnz;
        cuda_status = cudaMemcpy(matrix_destination, &s->dfdy[0],
                                 sizeof(double) * State::sparse_jac_nnz,
                                 cudaMemcpyHostToDevice);
        assert(cuda_status == cudaSuccess);

        double test[State::sparse_jac_nnz];
        cuda_status = cudaMemcpy(test, matrix_destination,
                                 sizeof(double) * State::sparse_jac_nnz,
                                 cudaMemcpyDeviceToHost);
        assert(cuda_status == cudaSuccess);
        for (int j = 0; j < State::sparse_jac_nnz; j++) std::cout << std::scientific << test[j] << " " << std::endl;
        
        // copy vector b
        double* vector_destination = system_b + i * State::neqs;
        cuda_status = cudaMemcpy(vector_destination, &s->bmat[0],
                                 sizeof(double) * State::neqs,
                                 cudaMemcpyHostToDevice);
        assert(cuda_status == cudaSuccess);

        cuda_status = cudaMemcpy(test, vector_destination,
                                 sizeof(double) * State::neqs,
                                 cudaMemcpyDeviceToHost);
        assert(cuda_status == cudaSuccess);
        for (int j = 0; j < State::neqs; j++) std::cout << std::scientific << test[j] << " " << std::endl;
        
        i++;
    }
}

void MultiState::solver_to_states(UnifiedVector<State*>& batched_states) {
    cudaError_t cuda_status = cudaSuccess;

    // copy the solutions into the batched states
    int i = 0;
    for (State* s : batched_states) {
        // copy solution x
        double* vector_source = system_x + i * State::neqs;
        cuda_status = cudaMemcpy(&s->dynext[0], vector_source,
                                 sizeof(double) * State::neqs,
                                 cudaMemcpyDeviceToHost);
        assert(cuda_status == cudaSuccess);

        double test[State::neqs];
        cuda_status = cudaMemcpy(test, vector_source,
                                 sizeof(double) * State::neqs,
                                 cudaMemcpyDeviceToHost);
        assert(cuda_status == cudaSuccess);
        for (int j = 0; j < State::neqs; j++) std::cout << std::scientific << test[j] << " " << std::endl;

        
        std::cout << "reporting after solver: " << i << std::endl;
        s->print_state();
        i++;
    }
}


void MultiState::matrix_solve(UnifiedVector<State*>& batched_states) {
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

    // load the batched state systems into the solver memory
    states_to_solver(batched_states);

    // do the solve
    std::cout << "workspace size = " << workspace_size << std::endl;
    std::cout << "internal size = " << internal_size << std::endl;    
    cusolver_status = cusolverSpDcsrqrsvBatched(cusolverHandle,
                                                State::neqs, // size per subsystem
                                                State::neqs, // size per subsystem
                                                State::sparse_jac_nnz,
                                                system_description,
                                                matrices_csr_values,
                                                matrices_csr_row_count_d,
                                                matrices_csr_col_index_d,
                                                system_b,
                                                system_x,
                                                static_cast<int>(batched_states.size()),
                                                info,
                                                workspace);
    cudaError_t cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

    // load the solutions into the state memory
    solver_to_states(batched_states);

    // set the state statuses to update the step
    for (State* s : batched_states) {
        s->status = step_update;
    }
}

void MultiState::report_error(UnifiedVector<State*>& batched_states) {
    for (State* s : batched_states) {
        s->report_error();
    }
}

void MultiState::advance(UnifiedVector<State*>& batched_states) {
    // Use the first state in the vector to determine what to do
    State* s = batched_states[0];
    if (s->status == step_solve) {
        matrix_solve(batched_states);
    } else if (s->status == integration_failed) {
        report_error(batched_states);
    } else {
        for (State* s : batched_states) {
            s->advance();
        }
    }
}
