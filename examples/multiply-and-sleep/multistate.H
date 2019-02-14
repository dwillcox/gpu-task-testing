#ifndef MULTISTATE_H
#define MULTISTATE_H
#include "state.H"
#include "unified_vector.H"

class MultiState {
public:
    MultiState(size_t max_states);

    ~MultiState();

    void advance(UnifiedVector<State*>&);
};
#endif
