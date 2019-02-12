#ifndef STATE_H
#define STATE_H
#include "generic_vector.h"

class State : public UnifiedMemoryClass {
public:
  double x;
  int counter;
  int status;

  State() : counter(0), status(0), x(0.0) {}

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
      counter++;
      if (counter == 3) status = 3;
      else status = 1;
    } else if (status == 1) {
      square();
      counter++;
      if (counter == 3) status = 3;
      else status = 2;      
    }
  }

  static void batched_advance(GenericVector<State*>& batched_states) {
    for (State* s : batched_states) {
      if (s->status == 2) {
	s->square();
	s->counter++;
	if (s->counter == 3) s->status = 3;
	else s->status = 0;
      }
    }
  }
};
#endif
