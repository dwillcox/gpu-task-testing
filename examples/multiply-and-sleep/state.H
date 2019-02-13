#ifndef STATE_H
#define STATE_H
#include "unified_vector.H"

class State : public UnifiedMemoryClass {
public:
  double x;
  int counter;
  int status;

  State() : counter(0), status(0), x(0.0) {}

  __device__
      void cube();

  __host__ __device__
      void square();

  __device__
      void advance();

  static void batched_advance(UnifiedVector<State*>&);
};
#endif
