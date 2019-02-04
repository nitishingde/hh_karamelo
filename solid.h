/* -*- c++ -*- ----------------------------------------------------------*/

#ifndef MPM_SOLID_H
#define MPM_SOLID_H

#include "pointers.h"
#include <vector>

class Solid : protected Pointers {
 public:
  string id;

  Solid(class MPM *, vector<string>);
  virtual ~Solid();
  virtual void init();
  void options(vector<string> *, vector<string>::iterator);
};

#endif
