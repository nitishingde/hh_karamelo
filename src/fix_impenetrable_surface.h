/* -*- c++ -*- ----------------------------------------------------------
 *
 *                    ***       Karamelo       ***
 *               Parallel Material Point Method Simulator
 *
 * Copyright (2019) Alban de Vaucorbeil, alban.devaucorbeil@monash.edu
 * Materials Science and Engineering, Monash University
 * Clayton VIC 3800, Australia

 * This software is distributed under the GNU General Public License.
 *
 * ----------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(impenetrablesurface, FixImpenetrableSurface)

#else

#ifndef MPM_FIX_IMPENETRABLE_SURFACE_H
#define MPM_FIX_IMPENETRABLE_SURFACE_H

#include "fix.h"
#include "var.h"
#include <vector>

class FixImpenetrableSurface : public Fix {
public:
  FixImpenetrableSurface(class MPM *, vector<string>);
  ~FixImpenetrableSurface();
  void setmask();
  void init();
  void setup();

  void initial_integrate();
  void post_particles_to_grid(){};
  void post_update_grid_state(){};
  void post_grid_to_point(){};
  void post_advance_particles(){};
  void post_velocities_to_grid(){};
  void final_integrate(){};

  void write_restart(ofstream *);
  void read_restart(ifstream *);

private:
  string usage = "Usage: fix(fix-ID, impenetrablesurface, group, K, xs, ys, "
                 "zs, nx, ny, nz)\n";
  int Nargs = 10;

  class Var xs_x, xs_y, xs_z;                  //< Position of a point on the surface
  class Var nx, ny, nz;                        //< Normal to the plane
  double K;                                    //< Contact stiffness
};

#endif
#endif

