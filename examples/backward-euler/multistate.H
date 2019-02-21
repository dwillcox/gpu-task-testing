#ifndef MULTISTATE_H
#define MULTISTATE_H
#include <vector>
#include <fstream>
#include <cusolverSp.h>
#include <cusparse.h>
#include "state.H"
#include "unified_vector.H"

class MultiState {
    // Solver
    size_t workspace_size, internal_size;
    cusparseMatDescr_t system_description;
    cusolverSpHandle_t cusolverHandle;
    cusparseHandle_t cusparseHandle;
    csrqrInfo_t info;
    void*  workspace;

    // System
    int* matrices_csr_col_index_d;
    int* matrices_csr_row_count_d;
    int matrices_csr_col_index_h[State::sparse_jac_nnz];
    int matrices_csr_row_count_h[State::neqs+1];
    double* matrices_csr_values;
    double* system_x;
    double* system_b;

    // Solver call statistics
    std::vector<int> num_linear_systems;
public:
    MultiState(size_t max_states);

    ~MultiState();

    void write_statistics();

    void create_linear_solver(size_t);

    void destroy_linear_solver();

    void states_to_solver(UnifiedVector<State*>&);

    void solver_to_states(UnifiedVector<State*>&);

    void matrix_solve(UnifiedVector<State*>&);

    void report_error(UnifiedVector<State*>&);

    void check_weirdness(UnifiedVector<State*>&);

    void advance(UnifiedVector<State*>&);
};
#endif
