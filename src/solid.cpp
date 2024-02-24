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

#include "solid.h"
#include "domain.h"
#include "error.h"
#include "input.h"
#include "material.h"
#include "memory.h"
#include "method.h"
#include "mpm.h"
#include "mpm_math.h"
#include "universe.h"
#include "update.h"
#include "var.h"
#include <Eigen/Eigen>
#include <math.h>
#include <mpi.h>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;
using namespace Eigen;
using namespace MPM_Math;


#define SQRT_3_OVER_2 1.224744871 // sqrt(3.0/2.0)
#define FOUR_THIRD 1.333333333333333333333333333333333333333

vector<string> split (string s, string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

Solid::Solid(MPM *mpm, vector<string> args) : Pointers(mpm)
{
  // Check that a method is available:
  if (update->method == nullptr)
  {
    error->all(FLERR, "Error: a method should be defined before creating a solid!\n");
  }

  if (args.size() < 2)
  {
    string error_str = "Error: solid command not enough arguments. ";
    for (auto &x : usage)
      error_str += x.second;
    error->all(FLERR, error_str);
  }

  id          = args[0];

  if (universe->me == 0)
    cout << "Creating new solid with ID: " << id << endl;

  method_type = update->method_type;

  np = 0;

  if (update->method->is_CPDI) {
    nc = pow(2, domain->dimension);
  }
  else
    nc = 0;

  mat = nullptr;

  if (update->method->is_TL) {
    is_TL = true;
    grid = new Grid(mpm);
  }
  else {
    is_TL = false;
    grid = domain->grid;
  }

  if (update->sub_method_type == Update::SubMethodType::APIC ||
      update->sub_method_type == Update::SubMethodType::AFLIP ||
      update->sub_method_type == Update::SubMethodType::ASFLIP) {
    apic = true;
  } else {
    apic = false;
  }

  dtCFL = 1.0e22;
  max_p_wave_speed = 0;
  vtot  = 0;
  mtot = 0;
  comm_n = 50; // Number of double to pack for particle exchange between CPUs.


  if (args[1].compare("restart") == 0) {
    // If the keyword restart, we are expecting to have read_restart()
    // launched right after.
    return;
  }

  if (usage.find(args[1]) == usage.end())
  {
    string error_str = "Error, keyword \033[1;31m" + args[1] + "\033[0m unknown!\n";
    for (auto &x : usage)
      error_str += x.second;
    error->all(FLERR, error_str);
  }

  if (args.size() < Nargs.find(args[1])->second)
  {
    error->all(FLERR, "Error: not enough arguments.\n"
	       + usage.find(args[1])->second);
  }

  if (args[1].compare("region") == 0)
  {
    // Set material, cellsize, and initial temperature:
    options(&args, args.begin() + 4);

    // Create particles:
    populate(args);
  }
  else if (args[1].compare("mesh") == 0)
  {
    // Set material and cellsize and initial temperature:
    options(&args, args.begin() + 3);

    read_mesh(args[2]);
  }
  else if (args[1].compare("file") == 0)
  {
    // Set material and cellsize and initial temperature:
    options(&args, args.begin() + 3);

    read_file(args[2]);
  }

  if (update->method->temp)
    comm_n = 54; // Number of double to pack for particle exchange between CPUs.
  else
    comm_n = 49;
}

Solid::~Solid()
{
  if (is_TL) delete grid;
}

void Solid::init()
{
  if (universe->me == 0) {
    cout << "Bounds for " << id << ":\n";
    cout << "xlo xhi: " << solidlo[0] << " " << solidhi[0] << endl;
    cout << "ylo yhi: " << solidlo[1] << " " << solidhi[1] << endl;
    cout << "zlo zhi: " << solidlo[2] << " " << solidhi[2] << endl;
  }

  // Calculate total volume:
  double vtot_local = 0;
  double mtot_local = 0;
  for (int ip=0; ip<np_local; ip++)
    {
      vtot_local += vol[ip];
      mtot_local += mass[ip];
  }

  MPI_Allreduce(&vtot_local, &vtot, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);
  MPI_Allreduce(&mtot_local, &mtot, 1, MPI_DOUBLE, MPI_SUM, universe->uworld);

  if (universe->me == 0) {
    cout << "Solid " << id << " total volume = " << vtot << endl;
    cout << "Solid " << id << " total mass = " << mtot << endl;
  }

  if (grid->nnodes == 0) grid->init(solidlo, solidhi);

  if (np == 0) {
    error->one(FLERR,"Error: solid does not have any particles.\n");
  }
}

void Solid::options(vector<string> *args, vector<string>::iterator it)
{
  // cout << "In solid::options()" << endl;
  if (args->end() < it + 3)
  {
    error->all(FLERR, "Error: not enough arguments.\n");
  }
  if (args->end() > it)
  {
    int iMat = material->find_material(*it);

    if (iMat == -1)
    {
      cout << "Error: could not find material named " << *it << endl;
      error->all(FLERR,"\n");
    }

    mat = &material->materials[iMat]; // point mat to the right material

    it++;

    grid->setup(*it); // set the grid cellsize

    it++;
    T0 = input->parsev(*it); // set initial temperature

    it++;

    if (it != args->end())
    {
      string error_str = "Error: too many arguments\n";
      for (auto &x : usage)
        error_str += x.second;
    }
  }
}

void Solid::grow(int nparticles)
{
  ptag.resize(nparticles);
  x0.resize(nparticles);
  x.resize(nparticles);

  v.resize(nparticles);
  v_update.resize(nparticles);
  a.resize(nparticles);
  mbp.resize(nparticles);
  f.resize(nparticles);
  sigma.resize(nparticles);
  strain_el.resize(nparticles);
  vol0PK1.resize(nparticles);
  L.resize(nparticles);
  F.resize(nparticles);
  R.resize(nparticles);
  D.resize(nparticles);
  Finv.resize(nparticles);
  Fdot.resize(nparticles);
  vol0.resize(nparticles);
  vol.resize(nparticles);
  rho0.resize(nparticles);
  rho.resize(nparticles);
  mass.resize(nparticles);
  eff_plastic_strain.resize(nparticles);
  eff_plastic_strain_rate.resize(nparticles);
  damage.resize(nparticles);
  damage_init.resize(nparticles);
  ienergy.resize(nparticles);
  mask.resize(nparticles);
  J.resize(nparticles);

  numneigh_pn.resize(nparticles);
  neigh_pn.resize(nparticles);
  wf_pn.resize(nparticles);
  if (nc != 0)
    wf_pn_corners.resize(nc * nparticles);
  wfd_pn.resize(nparticles);

  bigint nnodes = grid->nnodes_local + grid->nnodes_ghost;

  numneigh_np.resize(nnodes);
  neigh_np.resize(nnodes);
  wf_np.resize(nnodes);
  wfd_np.resize(nnodes);

  if (mat->cp != 0) {
    T.resize(nparticles);
    gamma.resize(nparticles);
    q.resize(nparticles);
  }
}

void Solid::compute_mass_nodes(bool reset)
{
  int ip;
  int nn = grid->nnodes_local + grid->nnodes_ghost;

  for (int in = 0; in < nn; in++)
    {
      if (reset) grid->mass[in] = 0;

      if (grid->rigid[in] && !mat->rigid) continue;

      for (int j = 0; j < numneigh_np[in]; j++)
	{
	  ip = neigh_np[in][j];
	  grid->mass[in] += wf_np[in][j] * mass[ip];
	}
    }
  return;
}

void Solid::compute_velocity_nodes(bool reset)
{
  Eigen::Vector3d vtemp, vtemp_update;
  //double mass_rigid;
  int ip;
  int nn = grid->nnodes_local + grid->nnodes_ghost;

  for(int in = 0; in < nn; in++) {
    if(reset) {
      grid->v[in].setZero();
      //grid->v_update[in].setZero();
      if(grid->rigid[in]) {
      	grid->mb[in].setZero();
      }
    }

    if(grid->rigid[in] && !mat->rigid) continue;

    if(grid->mass[in] > 0) {
      vtemp.setZero();
      if(grid->rigid[in])
    	  vtemp_update.setZero();

      for(int j = 0; j < numneigh_np[in]; j++) {
        ip = neigh_np[in][j];
	      if(grid->rigid[in]) {
	        vtemp_update += (wf_np[in][j] * mass[ip]) * v_update[ip];
	      }
	      if(update->method->ge) {
	        vtemp += (wf_np[in][j] * mass[ip]) * (v[ip] + L[ip] * (grid->x0[in] - x[ip]));
	      }
        else {
	        vtemp += wf_np[in][j] * mass[ip] * v[ip];
	      }
        // grid->v[in] += (wf_np[in][j] * mass[ip]) * v[ip]/ grid->mass[in];
      }
      vtemp /= grid->mass[in];
      grid->v[in] += vtemp;
      if(grid->rigid[in]) {
	      vtemp_update /= grid->mass[in];
	      grid->mb[in] += vtemp_update; // This should be grid->v_update[in], but we are using mb to make the reduction of ghost particles easy. It will be copied to grid->v_update[in] in Grid::update_grid_velocities()
      }
    }
  }
}

void Solid::compute_external_and_internal_forces_nodes_UL(bool reset)
{
  int ip;
  int nn = grid->nnodes_local + grid->nnodes_ghost;

  for (int in = 0; in < nn; in++) {
    if (reset) {
      grid->f[in].setZero();
      grid->mb[in].setZero();
    }

    if (grid->rigid[in]) {
      for (int j = 0; j < numneigh_np[in]; j++) {
        ip = neigh_np[in][j];
        grid->f[in] -= vol[ip] * (sigma[ip] * wfd_np[in][j]);
      }

      if (domain->axisymmetric == true) {
        for (int j = 0; j < numneigh_np[in]; j++) {
          ip = neigh_np[in][j];
          grid->f[in][0] -=
              vol[ip] * (sigma[ip](2, 2) * wf_np[in][j] / x[ip][0]);
        }
      }
    } else {
      for (int j = 0; j < numneigh_np[in]; j++) {
        ip = neigh_np[in][j];
        grid->f[in] -= vol[ip] * (sigma[ip] * wfd_np[in][j]);
        grid->mb[in] += wf_np[in][j] * mbp[ip];
      }

      if (domain->axisymmetric == true) {
        for (int j = 0; j < numneigh_np[in]; j++) {
          ip = neigh_np[in][j];
          grid->f[in][0] -=
              vol[ip] * (sigma[ip](2, 2) * wf_np[in][j] / x[ip][0]);
        }
      }
    }
  }
}

void Solid::compute_particle_accelerations_velocities_and_positions() {
  vector<Eigen::Vector3d> vc_update;
  vc_update.resize(nc);
  int in;
  double inv_dt = 1.0/update->dt;

  for(int ip = 0; ip < np_local; ip++) {
    v_update[ip].setZero();
    a[ip].setZero();

    for(int j = 0; j < numneigh_pn[ip]; j++) {
      in = neigh_pn[ip][j];
      v_update[ip] += wf_pn[ip][j] * grid->v_update[in];
      a[ip]        += wf_pn[ip][j] * (grid->v_update[in] - grid->v[in]);
    }
    a[ip] *= inv_dt;
    f[ip]  = a[ip] * mass[ip];
    x[ip] += update->dt * v_update[ip];

    if(!is_TL) {
      // Check if the particle is within the box's domain:
      if (domain->inside(x[ip]) == 0) {
        cout << "Error: Particle " << ip << " left the domain ("
             << domain->boxlo[0] << "," << domain->boxhi[0] << ","
             << domain->boxlo[1] << "," << domain->boxhi[1] << ","
             << domain->boxlo[2] << "," << domain->boxhi[2] << ",):\n"
             << x[ip] << endl;
        error->one(FLERR, "");
      }
    }
  }
}

void Solid::update_particle_velocities(double alpha) {
  for(int ip = 0; ip < np_local; ip++) {
    v[ip] = (1 - alpha) * v_update[ip] + alpha * (v[ip] + update->dt * a[ip]);
  }
}

void Solid::compute_rate_deformation_gradient_UL(bool doublemapping)
{
  if (mat->rigid)
    return;

  int in;
  vector<Eigen::Vector3d> *vn;

  if (doublemapping)
    vn = &grid->v;
  else
    vn = &grid->v_update;

  for(int ip = 0; ip < np_local; ip++) {
	  L[ip].setZero();
	  for(int j = 0; j < numneigh_pn[ip]; j++) {
      in = neigh_pn[ip][j];
      L[ip](0,0) += (*vn)[in][0]*wfd_pn[ip][j][0];
      L[ip](0,1) += (*vn)[in][0]*wfd_pn[ip][j][1];
      L[ip](0,2) += (*vn)[in][0]*wfd_pn[ip][j][2];
      L[ip](1,0) += (*vn)[in][1]*wfd_pn[ip][j][0];
      L[ip](1,1) += (*vn)[in][1]*wfd_pn[ip][j][1];
      L[ip](1,2) += (*vn)[in][1]*wfd_pn[ip][j][2];
      L[ip](2,0) += (*vn)[in][2]*wfd_pn[ip][j][0];
      L[ip](2,1) += (*vn)[in][2]*wfd_pn[ip][j][1];
      L[ip](2,2) += (*vn)[in][2]*wfd_pn[ip][j][2];
    }
  }
}

void Solid::update_deformation_gradient()
{
  if (mat->rigid)
    return;

  bool nh;
  Eigen::Matrix3d eye;
  eye.setIdentity();

  if (mat->type == material->constitutive_model::NEO_HOOKEAN)
    nh = true;
  else
    nh = false;

  for(int ip = 0; ip < np_local; ip++) {
    F[ip]    = (eye + update->dt * L[ip]) * F[ip];
    Finv[ip] = F[ip].inverse();
    J[ip]    = F[ip].determinant();
    vol[ip]  = J[ip] * vol0[ip];

    if(J[ip] <= 0.0 && damage[ip] < 1.0) {
      cout << "Error: J[" << ptag[ip] << "]<=0.0 == " << J[ip] << endl;
      cout << "F[" << ptag[ip] << "]:" << endl << F[ip] << endl;
      cout << "Fdot[" << ptag[ip] << "]:" << endl << Fdot[ip] << endl;
      cout << "damage[" << ptag[ip] << "]:" << endl << damage[ip] << endl;
      error->one(FLERR,"");
    }
    rho[ip] = rho0[ip] / J[ip];

    if(!nh) {
      // Only done if not Neo-Hookean:
      D[ip] = 0.5 * (L[ip] + L[ip].transpose());
    }
  }
}

void Solid::update_stress()
{
  if (mat->rigid)
    return;

  max_p_wave_speed = 0;
  double flow_stress;
  Matrix3d eye;
  eye.setIdentity();

  vector<double> pH(np_local, 0);
  vector<double> plastic_strain_increment(np_local, 0);
  vector<Eigen::Matrix3d> sigma_dev;
  sigma_dev.resize(np_local);
  double tav = 0;

  for(int ip = 0; ip < np_local; ip++) {
    if(mat->cp != 0) {
      mat->eos->compute_pressure(pH[ip], ienergy[ip], J[ip], rho[ip], damage[ip], D[ip], grid->cellsize, T[ip]);
      pH[ip] += mat->temp->compute_thermal_pressure(T[ip]);
      sigma_dev[ip] = mat->strength->update_deviatoric_stress(sigma[ip], D[ip], plastic_strain_increment[ip], eff_plastic_strain[ip], eff_plastic_strain_rate[ip], damage[ip], T[ip]);
    }
    else {
      mat->eos->compute_pressure(pH[ip], ienergy[ip], J[ip], rho[ip], damage[ip], D[ip], grid->cellsize);
      sigma_dev[ip] = mat->strength->update_deviatoric_stress(sigma[ip], D[ip], plastic_strain_increment[ip], eff_plastic_strain[ip], eff_plastic_strain_rate[ip], damage[ip]);
    }

    eff_plastic_strain[ip] += plastic_strain_increment[ip];

    // // compute a characteristic time over which to average the plastic
    // strain

    tav = 1000 * grid->cellsize / mat->signal_velocity;

    eff_plastic_strain_rate[ip] -= eff_plastic_strain_rate[ip] * update->dt / tav;
    eff_plastic_strain_rate[ip] += plastic_strain_increment[ip] / tav;
    eff_plastic_strain_rate[ip] = MAX(0.0, eff_plastic_strain_rate[ip]);

    if(mat->damage != nullptr) {
	    mat->damage->compute_damage(damage_init[ip], damage[ip], pH[ip], sigma_dev[ip], eff_plastic_strain_rate[ip], plastic_strain_increment[ip]);
    }

    if(mat->cp != 0) {
      flow_stress = SQRT_3_OVER_2 * sigma_dev[ip].norm();
      mat->temp->compute_heat_source(T[ip], gamma[ip], flow_stress, eff_plastic_strain_rate[ip]);
      gamma[ip] *= vol[ip] * mat->invcp;
    }

    if(damage[ip] == 0 || pH[ip] >= 0)
	    sigma[ip] = -pH[ip] * eye + sigma_dev[ip];
    else
	    sigma[ip] = -pH[ip] * (1.0 - damage[ip])* eye + sigma_dev[ip];

    if(damage[ip] > 1e-10) {
      strain_el[ip] = (update->dt * D[ip].trace() + strain_el[ip].trace()) / 3.0 * eye + sigma_dev[ip] / (mat->G * (1 - damage[ip]));
    }
    else {
      strain_el[ip] = (update->dt * D[ip].trace() + strain_el[ip].trace()) / 3.0 * eye + sigma_dev[ip] / mat->G;
    }
  }

  double min_h_ratio = 1.0;

  for(int ip = 0; ip < np_local; ip++) {
    if (damage[ip] >= 1.0)
      continue;

    max_p_wave_speed = MAX(max_p_wave_speed, sqrt((mat->K + FOUR_THIRD * mat->G) / rho[ip]) + MAX(MAX(fabs(v[ip](0)), fabs(v[ip](1))), fabs(v[ip](2))));

    if(std::isnan(max_p_wave_speed)) {
      cout << "Error: max_p_wave_speed is nan with ip=" << ip
           << ", ptag[ip]=" << ptag[ip] << ", rho0[ip]=" << rho0[ip]<< ", rho[ip]=" << rho[ip]
           << ", K=" << mat->K << ", G=" << mat->G << ", J[ip]=" << J[ip]
           << endl;
      error->one(FLERR, "");
    }
    else if(max_p_wave_speed < 0.0) {
      cout << "Error: max_p_wave_speed= " << max_p_wave_speed
           << " with ip=" << ip << ", rho[ip]=" << rho[ip] << ", K=" << mat->K
           << ", G=" << mat->G << endl;
      error->one(FLERR, "");
    }
  }

  dtCFL = MIN(dtCFL, grid->cellsize * min_h_ratio / max_p_wave_speed);

  if(std::isnan(dtCFL)) {
    cout << "Error: dtCFL = " << dtCFL << "\n";
    cout << "max_p_wave_speed = " << max_p_wave_speed
         << ", grid->cellsize=" << grid->cellsize << endl;
    error->one(FLERR, "");
  }
}

void Solid::copy_particle(int i, int j) {
  ptag[j]                    = ptag[i];
  x0[j]                      = x0[i];
  x[j]                       = x[i];
  v[j]                       = v[i];
  v_update[j]                = v_update[i];
  a[j]                       = a[i];
  mbp[j]                     = mbp[i];
  f[j]                       = f[i];
  vol0[j]                    = vol0[i];
  vol[j]                     = vol[i];
  rho0[j]                    = rho0[i];
  rho[j]                     = rho[i];
  mass[j]                    = mass[i];
  eff_plastic_strain[j]      = eff_plastic_strain[i];
  eff_plastic_strain_rate[j] = eff_plastic_strain_rate[i];
  damage[j]                  = damage[i];
  damage_init[j]             = damage_init[i];
  if (update->method->temp) {
    T[j]                     = T[i];
    gamma[j]                   = gamma[i];
    q[j]                       = q[i];
  }
  ienergy[j]                 = ienergy[i];
  mask[j]                    = mask[i];
  sigma[j]                   = sigma[i];
  strain_el[j]               = strain_el[i];
  vol0PK1[j]                 = vol0PK1[i];
  L[j]                       = L[i];
  F[j]                       = F[i];
  R[j]                       = R[i];
  D[j]                       = D[i];
  Finv[j]                    = Finv[i];
  Fdot[j]                    = Fdot[i];
  J[j]                       = J[i];
}

void Solid::pack_particle(int i, vector<double> &buf) {
  buf.push_back(ptag[i]);

  buf.push_back(x[i](0));
  buf.push_back(x[i](1));
  buf.push_back(x[i](2));

  buf.push_back(x0[i](0));
  buf.push_back(x0[i](1));
  buf.push_back(x0[i](2));

  buf.push_back(v[i](0));
  buf.push_back(v[i](1));
  buf.push_back(v[i](2));

  buf.push_back(v_update[i](0));
  buf.push_back(v_update[i](1));
  buf.push_back(v_update[i](2));

  buf.push_back(a[i](0));
  buf.push_back(a[i](1));
  buf.push_back(a[i](2));

  buf.push_back(mbp[i](0));
  buf.push_back(mbp[i](1));
  buf.push_back(mbp[i](2));

  buf.push_back(f[i](0));
  buf.push_back(f[i](1));
  buf.push_back(f[i](2));

  buf.push_back(vol0[i]);
  buf.push_back(vol[i]);
  
  buf.push_back(rho0[i]);
  buf.push_back(rho[i]);

  buf.push_back(mass[i]);
  buf.push_back(eff_plastic_strain[i]);
  buf.push_back(eff_plastic_strain_rate[i]);
  buf.push_back(damage[i]);
  buf.push_back(damage_init[i]);
  if (update->method->temp) {
    buf.push_back(T[i]);
    buf.push_back(gamma[i]);
    buf.push_back(q[i][0]);
    buf.push_back(q[i][1]);
    buf.push_back(q[i][2]);
  }
  buf.push_back(ienergy[i]);
  buf.push_back(mask[i]);

  buf.push_back(sigma[i](0,0));
  buf.push_back(sigma[i](1,1));
  buf.push_back(sigma[i](2,2));
  buf.push_back(sigma[i](0,1));
  buf.push_back(sigma[i](0,2));
  buf.push_back(sigma[i](1,2));

  buf.push_back(F[i](0,0));
  buf.push_back(F[i](0,1));
  buf.push_back(F[i](0,2));
  buf.push_back(F[i](1,0));
  buf.push_back(F[i](1,1));
  buf.push_back(F[i](1,2));
  buf.push_back(F[i](2,0));
  buf.push_back(F[i](2,1));
  buf.push_back(F[i](2,2));

  buf.push_back(J[i]);
}

void Solid::unpack_particle(int &i, vector<int> list, vector<double> &buf) {
  int m;
  for(auto j: list) {
    m = j;

    ptag[i] = (tagint) buf[m++];

    x[i](0) = buf[m++];
    x[i](1) = buf[m++];
    x[i](2) = buf[m++];

    x0[i](0) = buf[m++];
    x0[i](1) = buf[m++];
    x0[i](2) = buf[m++];

    v[i](0) = buf[m++];
    v[i](1) = buf[m++];
    v[i](2) = buf[m++];

    v_update[i](0) = buf[m++];
    v_update[i](1) = buf[m++];
    v_update[i](2) = buf[m++];

    a[i](0) = buf[m++];
    a[i](1) = buf[m++];
    a[i](2) = buf[m++];

    mbp[i](0) = buf[m++];
    mbp[i](1) = buf[m++];
    mbp[i](2) = buf[m++];

    f[i](0) = buf[m++];
    f[i](1) = buf[m++];
    f[i](2) = buf[m++];

    vol0[i] = buf[m++];
    vol[i] = buf[m++];

    rho0[i] = buf[m++];
    rho[i] = buf[m++];

    mass[i] = buf[m++];
    eff_plastic_strain[i] = buf[m++];
    eff_plastic_strain_rate[i] = buf[m++];
    damage[i] = buf[m++];
    damage_init[i] = buf[m++];
    if(update->method->temp) {
      T[i] = buf[m++];
      gamma[i] = buf[m++];

      q[i][0] = buf[m++];
      q[i][1] = buf[m++];
      q[i][2] = buf[m++];
    }

    ienergy[i] = buf[m++];
    mask[i] = buf[m++];

    sigma[i](0,0) = buf[m++];
    sigma[i](1,1) = buf[m++];
    sigma[i](2,2) = buf[m++];
    sigma[i](0,1) = buf[m++];
    sigma[i](0,2) = buf[m++];
    sigma[i](1,2) = buf[m++];
    sigma[i](1,0) = sigma[i](0,1);
    sigma[i](2,0) = sigma[i](0,2);
    sigma[i](2,1) = sigma[i](1,2);

    F[i](0,0) = buf[m++];
    F[i](0,1) = buf[m++];
    F[i](0,2) = buf[m++];
    F[i](1,0) = buf[m++];
    F[i](1,1) = buf[m++];
    F[i](1,2) = buf[m++];
    F[i](2,0) = buf[m++];
    F[i](2,1) = buf[m++];
    F[i](2,2) = buf[m++];

    J[i] = buf[m++];
    i++;
  }
}

void Solid::populate(vector<string> args) {
  if(universe->me == 0) {
    cout << "Solid delimitated by region ID: " << args[2] << endl;
  }

  // Look for region ID:
  int iregion = domain->find_region(args[2]);
  if(iregion == -1) {
    error->all(FLERR, "Error: region ID " + args[2] + " not does not exist.\n");
  }

  if(domain->created==false) {
    error->all(FLERR, "The domain must be created before any solids can (create_domain(...)).");
  }

  double *sublo = domain->sublo;
  double *subhi = domain->subhi;

  vector<double> limits = domain->regions[iregion]->limits();

  solidlo[0] = limits[0];
  solidhi[0] = limits[1];
  solidlo[1] = limits[2];
  solidhi[1] = limits[3];
  solidlo[2] = limits[4];
  solidhi[2] = limits[5];

  solidsublo[0] = MAX(solidlo[0], sublo[0]);
  solidsublo[1] = MAX(solidlo[1], sublo[1]);
  solidsublo[2] = MAX(solidlo[2], sublo[2]);

  solidsubhi[0] = MIN(solidhi[0], subhi[0]);
  solidsubhi[1] = MIN(solidhi[1], subhi[1]);
  solidsubhi[2] = MIN(solidhi[2], subhi[2]);

#ifdef DEBUG
  cout << "proc " << universe->me
       << "\tsolidsublo=[" << solidsublo[0] << "," << solidsublo[1] << "," << solidsublo[2]
       << "]\t solidsubhi=["<< solidsubhi[0] << "," << solidsubhi[1] << "," << solidsubhi[2]
       << "]\n";

//   std::vector<double> x2plot, y2plot;
#endif

  // Calculate total number of particles np_local:
  int nsubx, nsuby, nsubz;
  double delta;

  double *boundlo = domain->boxlo, *boundhi = domain->boxhi;
  delta = grid->cellsize;
  double Loffsetlo[3] = {MAX(0.0, sublo[0] - boundlo[0]),
                         MAX(0.0, sublo[1] - boundlo[1]),
                         MAX(0.0, sublo[2] - boundlo[2])};
  double Loffsethi[3] = {MAX(0.0, MIN(subhi[0], boundhi[0]) - boundlo[0]),
                         MAX(0.0, MIN(subhi[1], boundhi[1]) - boundlo[1]),
                         MAX(0.0, MIN(subhi[2], boundhi[2]) - boundlo[2])};

  int noffsetlo[3] = {(int)floor(Loffsetlo[0] / delta),
                      (int)floor(Loffsetlo[1] / delta),
                      (int)floor(Loffsetlo[2] / delta)};

  int noffsethi[3] = {(int)ceil(Loffsethi[0] / delta),
                      (int)ceil(Loffsethi[1] / delta),
                      (int)ceil(Loffsethi[2] / delta)};

  cout << "1--- proc " << universe->me << " noffsetlo=[" << noffsetlo[0]
       << "," << noffsetlo[1] << "," << noffsetlo[2] << "]\n";
  cout << "1--- proc " << universe->me << " noffsethi=[" << noffsethi[0]
       << "," << noffsethi[1] << "," << noffsethi[2] << "]\n";

    // cout << "abs=" << abs(boundlo[0] + noffsethi[0] * delta - subhi[0])<< "]\n";
  if (universe->procneigh[0][1] >= 0 &&
      abs(boundlo[0] + noffsethi[0] * delta - subhi[0]) < 1.0e-12) {
    noffsethi[0]++;
  }
  if (domain->dimension >= 2 && universe->procneigh[1][1] >= 0 &&
      abs(boundlo[1] + noffsethi[1] * delta - subhi[1]) < 1.0e-12) {
    noffsethi[1]++;
  }
  if (domain->dimension == 3 && universe->procneigh[2][1] >= 0 &&
      abs(boundlo[2] + noffsethi[2] * delta - subhi[2]) < 1.0e-12) {
      noffsethi[2]++;
    }

  cout << "2--- proc " << universe->me << " noffsethi=[" << noffsethi[0]
         << "," << noffsethi[1] << "," << noffsethi[2] << "]\n";

  nsubx = MAX(0, noffsethi[0] - noffsetlo[0]);
  if(domain->dimension >= 2) {
    nsuby = MAX(0, noffsethi[1] - noffsetlo[1]);
  } else {
    nsuby = 1;
  }
  if(domain->dimension >= 3) {
    nsubz = MAX(0,noffsethi[2] - noffsetlo[2]);
  } else {
    nsubz = 1;
  }

  if (universe->procneigh[0][1] == -1) {
    while (boundlo[0] + delta * (noffsetlo[0] + nsubx - 0.5) <
           MIN(subhi[0], boundhi[0]))
      nsubx++;
  }
  if (universe->procneigh[1][1] == -1) {
    while (boundlo[1] + delta * (noffsetlo[1] + nsuby - 0.5) <
           MIN(subhi[1], boundhi[1]))
      nsuby++;
  }
  if (universe->procneigh[2][1] == -1) {
    while (boundlo[2] + delta * (noffsetlo[2] + nsubz - 0.5) <
           MIN(subhi[2], boundhi[2]))
      nsubz++;
  }
  cout << "2--proc " << universe->me << "\t nsub=[" << nsubx << "," << nsuby
       << "," << nsubz << "]\n";

  np_local = nsubx * nsuby * nsubz;

  // Create particles:
  if(universe->me == 0)
    cout << "delta = " << delta << endl;

  int l = 0;
  double vol_;

  if(domain->dimension == 1)
    vol_ = delta;
  else if(domain->dimension == 2)
    vol_ = delta * delta;
  else
    vol_ = delta * delta * delta;

  double mass_;
  if (mat->rigid)
    mass_ = 1;
  else
    mass_ = mat->rho0 * vol_;

  np_per_cell = (int) input->parsev(args[3]);
  double xi = 0.5;
  double lp = delta;
  int nip   = 1;
  vector<double> intpoints;

  if (np_per_cell == 1)
  {
    // One particle per cell at the center:

    xi = 0.5;
    lp *= 0.5;
    nip = 1;

    intpoints = {0, 0, 0};
  }
  else if (np_per_cell == 2)
  {
    // Quadratic elements:

    if      (domain->dimension == 1) nip = 2;
    else if (domain->dimension == 2) nip = 4;
    else                             nip = 8;

    xi = 0.25;
    lp *= 0.25;
    intpoints = {-xi, -xi, -xi, -xi, xi, -xi, xi, -xi, -xi, xi, xi, -xi,
                 -xi, -xi, xi,  -xi, xi, xi,  xi, -xi, xi,  xi, xi, xi};
  }
  else if (np_per_cell == 3)
  {
    // Berstein elements:

    if (nc == 0)
      xi = 0.7746 / 2;
    else
      xi = 1.0 / 3.0;

    lp *= 1.0 / 6.0;
    nip = 27;

    if (domain->dimension == 1) nip = 3;
    else if (domain->dimension == 2) nip = 9;
    else nip = 27;

    intpoints = {-xi, -xi, -xi,
		 -xi, 0, -xi,
		 -xi, xi, -xi,
		 0, -xi, -xi,
		 0, 0, -xi,
		 0, xi, -xi,
		 xi, -xi, -xi,
		 xi, 0, -xi,
		 xi, xi, -xi,
		 -xi, -xi, 0,
		 -xi, 0, 0,
		 -xi, xi, 0,
		 0, -xi, 0,
		 0, 0, 0,
		 0, xi, 0,
		 xi, -xi, 0,
		 xi, 0, 0,
		 xi, xi, 0,
		 -xi, -xi, xi,
		 -xi, 0, xi,
		 -xi, xi, xi,
		 0, -xi, xi,
		 0, 0, xi,
		 0, xi, xi,
		 xi, -xi, xi,
		 xi, 0, xi,
		 xi, xi, xi};

  } else {
    lp *= 1.0 / (2 * np_per_cell);

    if (domain->dimension == 1) {
      nip = np_per_cell;
    } else if (domain->dimension == 2) {
      nip = np_per_cell * np_per_cell;
    } else {
      nip = np_per_cell * np_per_cell * np_per_cell;
    }

    double d = 1.0 / np_per_cell;

    for (int k = 0; k < np_per_cell; k++) {
      for (int i = 0; i < np_per_cell; i++) {
	for (int j = 0; j < np_per_cell; j++) {
	  intpoints.push_back((i + 0.5) * d - 0.5);
	  intpoints.push_back((j + 0.5) * d - 0.5);
	  intpoints.push_back((k + 0.5) * d - 0.5);
	}
      }
    }
  }

  np_local *= nip;

  mass_ /= (double) nip;
  vol_ /= (double) nip;

  // Allocate the space in the vectors for np particles:
  grow(np_local);

  int dim = domain->dimension;

  for (int i = 0; i < nsubx; i++)
    {
      for (int j = 0; j < nsuby; j++)
	{
	  for (int k = 0; k < nsubz; k++)
	    {
	      for (int ip = 0; ip < nip; ip++)
		{

		  if (l >= np_local)
		    {
		      cout << "Error in Solid::populate(), exceeding the allocated number of particles.\n";
		      cout << "l = " << l << endl;
		      cout << "np_local = " << np_local << endl;
		      error->all(FLERR, "");
		    }

		  x0[l][0] = x[l][0] =
		    boundlo[0] + delta*(noffsetlo[0] + i + 0.5 + intpoints[3*ip+0]);
		  x0[l][1] = x[l][1] =
		    boundlo[1] + delta*(noffsetlo[1] + j + 0.5 + intpoints[3*ip+1]);
		  if (dim == 3)
		    x0[l][2] = x[l][2] =
		      boundlo[2] + delta*(noffsetlo[2] + k + 0.5 + intpoints[3*ip+2]);
		  else
		    x0[l][2] = x[l][2] = 0;

		  // Check if the particle is inside the region:
		  if (domain->inside_subdomain(x0[l][0], x0[l][1], x0[l][2]) && domain->regions[iregion]->inside(x0[l][0], x0[l][1], x0[l][2]) == 1)
		    {
		      l++;
		    }
		}
	    }
	}
    }

  if (np_local > l)
    {
      grow(l);
    }
  np_local = l; // Adjust np_local to account for the particles outside the domain

  tagint ptag0 = 0;

  for (int proc=0; proc<universe->nprocs; proc++){
    int np_local_bcast;
    if (proc == universe->me) {
      // Send np_local
      np_local_bcast = np_local;
    } else {
      // Receive np_local
      np_local_bcast = 0;
    }
    MPI_Bcast(&np_local_bcast,1,MPI_INT,proc,universe->uworld);
    if (universe->me > proc) ptag0 += np_local_bcast;
  }

  np_local = l; // Adjust np to account for the particles outside the domain
  cout << "np_local=" << np_local << endl;

  for (int i = 0; i < np_local; i++)
  {
    a[i].setZero();
    v[i].setZero();
    f[i].setZero();
    mbp[i].setZero();
    v_update[i].setZero();
    rho0[i] = rho[i] = mat->rho0;

    if (domain->axisymmetric == true) {
      mass[i] = mass_ * x0[i][0];
      vol0[i] = vol[i] = mass[i] / rho0[i];
    } else {
      mass[i] = mass_;
      vol0[i] = vol[i] = vol_;
    }

    eff_plastic_strain[i]      = 0;
    eff_plastic_strain_rate[i] = 0;
    damage[i]                  = 0;
    damage_init[i]             = 0;
    if (update->method->temp) {
      T[i]                     = T0;
      gamma[i]                 = 0;
      q[i].setZero();
    }
    ienergy[i]                 = 0;
    strain_el[i].setZero();
    sigma[i].setZero();
    vol0PK1[i].setZero();
    L[i].setZero();
    F[i].setIdentity();
    R[i].setIdentity();
    D[i].setZero();
    Finv[i].setZero();
    Fdot[i].setZero();
    J[i] = 1;
    mask[i] = 1;

    ptag[i] = ptag0 + i + 1 + domain->np_total;
  }

  if (l != np_local)
  {
    cout << "Error l=" << l << " != np_local=" << np_local << endl;
    error->one(FLERR, "");
  }

  int np_local_reduced;
  MPI_Allreduce(&np_local, &np_local_reduced, 1, MPI_INT, MPI_SUM, universe->uworld);
  np += np_local_reduced;
  domain->np_total += np;
  domain->np_local += np_local;
}
