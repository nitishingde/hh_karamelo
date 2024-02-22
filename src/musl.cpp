/* ----------------------------------------------------------------------
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

#include "musl.h"
#include <iostream>
#include <vector>
#include "scheme.h"
#include "method.h"
#include "update.h"
#include "output.h"
#include "modify.h"
#include "universe.h"

using namespace std;

MUSL::MUSL(MPM *mpm) : Scheme(mpm) {
  // cout << "In MUSL::MUSL()" << endl;
}

void MUSL::setup() {
  output->setup();
}

void MUSL::run(Var condition) {
  bigint ntimestep = update->ntimestep;
  output->write(ntimestep);

  //for(int i=0; i<nsteps; i++){
  while((bool) condition.result(mpm)) {
    ntimestep = update->update_timestep();

    update->method->compute_grid_weight_functions_and_gradients();
    update->method->reset();
    modify->initial_integrate();
    update->method->particles_to_grid();//@MPI MPI_Allreduce 1 element, check for rigid solids
    update->method->update_grid_state();
    modify->post_update_grid_state();//@MPI MPI_Allreduce in FixVelocityNodes, Vector3 forceTotal
    update->method->grid_to_points();
    update->method->advance_particles();
    update->method->velocities_to_grid();
    modify->post_velocities_to_grid();
    update->method->compute_rate_deformation_gradient(true);
    update->method->update_deformation_gradient();
    update->method->update_stress(true);
    update->method->exchange_particles();//@MPI: bunch of mpi sends and recvs
    update->update_time();
    update->method->adjust_dt();//@MPI MPI_Allreduce 1 element, dtCFL_reduced for calculating the next time step (dt)

    if((update->maxtime != -1) and (update->atime > update->maxtime)) {
      update->nsteps = ntimestep;
      output->write(ntimestep);
      break;
    }

    if(ntimestep == output->next or ntimestep == update->nsteps) {
      output->write(ntimestep);
    }
  }
}

