/* -*- c++ -*- ----------------------------------------------------------*/


#ifndef LMP_DOMAIN_H
#define LMP_DOMAIN_H

#include <math.h>
#include "pointers.h"
#include "region.h"
#include "solid.h"
#include <map>
#include <string>
#include <vector>

using namespace std;

class Domain : protected Pointers {
 public:
  int dimension;                         // 2 = 2d, 3 = 3d
  vector<class Region *> regions;         // list of defined Regions
  vector<class Solid *> solids;         // list of defined Regions

  Domain(class MPM *);
  virtual ~Domain();

  void add_region(vector<string>);
  int find_region(string);

  typedef Region *(*RegionCreator)(MPM *,vector<string>);
  typedef map<string,RegionCreator> RegionCreatorMap;
  RegionCreatorMap *region_map;

 private:
  template <typename T> static Region *region_creator(MPM *,vector<string>);
};

#endif
