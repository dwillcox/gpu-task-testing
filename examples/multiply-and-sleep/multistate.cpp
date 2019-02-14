#include "multistate.H"

MultiState::MultiState(size_t max_states) {}

MultiState::~MultiState() {}

void MultiState::advance(UnifiedVector<State*>& batched_states) {
    for (State* s : batched_states) {
        if (s->status == 2) {
            s->square();
            s->counter++;
            if (s->counter == 3) s->status = 3;
            else s->status = 0;
        }
    }    
}
