#ifndef STATE_H
#define STATE_H
#include "generic_vector.h"

class State : public UnifiedMemoryClass {
public:
  double x;
  int status;

  __device__
  void cube() {
    x = x * x * x;
  }

  __host__ __device__
  void square() {
    x = x * x;
  }

  __device__
  void advance() {
    if (status == 0) {
      cube();
      status = 1;
    } else if (status == 1) {
      square();
      status = 2;
    }
  }

  static void batched_advance(GenericVector<State*>& batched_states) {
    for (State* s : batched_states) {
      if (s->status == 2) {
	s->square();
	s->status = 3;
      }
    }
  }
};
#endif
