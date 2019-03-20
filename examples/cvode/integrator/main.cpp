#include <iostream>
#include <cassert>
#include <AMReX.H>
#include <AMReX_REAL.H>
#include <AMReX_CONSTANTS.H>
#include "Vode.H"
#include "RealVector.H"

using namespace amrex;

int main(int argc, char* argv[]) {
    const size_t N = 10;

    Real tstart = zero;
    Real tend   = one;
    RealVector<N> start_vector;

    for (size_t i = 0; i < N; i++) start_vector[i] = i * one;

    std::cout << "start vector: " << std::endl;
    for (auto& x : start_vector) std::cout << x << " ";
    std::cout << std::endl;

    Vode<N> vode;

    vode.initialize(tstart, start_vector, tend, VodeStepping::Normal);

    RealVector<N> increment;
    for (size_t i = 0; i < N; i++) increment[i] = one;

    start_vector + increment;

    std::cout << "incremented start vector: " << std::endl;
    for (auto& x : start_vector) std::cout << x << " ";
    std::cout << std::endl;
    

    return 0;
}
