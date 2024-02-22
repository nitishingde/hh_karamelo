#include <iosfwd>
#include <hedgehog/hedgehog.h>
#include <mpi.h>

#include "error.h"
#include "mpm.h"
#include "input.h"
#include "method.h"
#include "modify.h"
#include "output.h"
#include "update.h"
#include "scheme.h"
#include "universe.h"
#include "var.h"
#include "ulmpm.h"
#include "mpi_wrappers.h"

constexpr bool LOG_INFO = false;

int extract_first_no(const std::string &str) {
  int no = 0;
  bool found = false;
  for(const auto c: str) {
    if('0' <= c and c <= '9') {
      found = true;
      no = 10*no + static_cast<int>(c-'0');
    }
    else if(found) {
      return no;
    }
  }

  return no;
}

struct HHContext {
  int32_t ntimestep = 0;
  Var     condition = {};
  MPM     *pMpm     = nullptr;
  // Update  *pUpdate  = nullptr;
  // Modify  *pModify  = nullptr;
  // Output  *pOutput  = nullptr;
  // Error   *pError   = nullptr;
  bool    breakLoop = false;
  float   dt        = 0.;
};

class InputTask: public hh::AbstractTask<2, MPM, HHContext, HHContext, float> {
public:
  explicit InputTask(hh::Graph<1, MPM, float> &graph): hh::AbstractTask<2, MPM, HHContext, HHContext, float>("InputTask", 1, false) {
    pGraph = &graph;
  }

  void execute(std::shared_ptr<MPM> mpm) override {
    if constexpr(LOG_INFO) printf("[%s:%d]\n", __FUNCTION__, __LINE__);
    auto ctx = std::make_shared<HHContext>();
    ctx->pMpm    = mpm.get();

    std::istream is(mpm->input->getInputFile());
    for(char str[256]; is.getline(str, 256, '\n');) {
      for(char &c: str) {
        if(c == '#') {
          c = '\0';
          break;
        }
      }
      //if(std::strlen(str) == 0) continue;
      lines_.emplace_back(str);
    }
    int size = lines_.size();
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    lines_.resize(size);

    this->process_ctx(ctx);
  }

  void execute(std::shared_ptr<HHContext> ctx) override {
    if(!static_cast<bool>(ctx->condition.result(ctx->pMpm)) or ctx->breakLoop) {
      this->process_ctx(ctx);
      return;
    }
    this->addResult(ctx);
  }

  [[nodiscard]] bool canTerminate() const override {
    return canTerminate_.load();
  }

private:
  void process_ctx(const std::shared_ptr<HHContext> &ctx) {
    auto input        = ctx->pMpm->input;
    auto &line_number = input->line_number;
    auto &line        = input->line;

    auto universe = ctx->pMpm->universe;
    auto me = universe->me;

    for(; line_number < lines_.size(); line_number++) {
      if(me == 0) {
        line = lines_[line_number];
      }

      MPI_string_bcast(line, MPI_CHAR, 0, universe->uworld);
      if(line == "quit") break;
      if(line.find("run") != std::string::npos) {
        run(ctx, extract_first_no(line));
        line_number++;
        return;
      }
      input->parsev(line).result();

      line.clear();

      int end = ((line_number+1) == lines_.size());
      MPI_Bcast(&end, 1, MPI_INT, 0, universe->uworld);
    }
    canTerminate_.store(true);
    this->addResult(std::make_shared<float>(42.));
  }

  void run(const std::shared_ptr<HHContext> &ctx, const int32_t nsteps) {
    auto update = ctx->pMpm->update;
    auto error  = ctx->pMpm->error;
    auto mpm    = ctx->pMpm;
    mpm->init();

    update->scheme->setup();

    // Check that a method is available:
    if(update->method == nullptr) {
      error->all(FLERR, "Error: no method was defined!\n");
    }

    update->nsteps    = nsteps;//INT_MAX;
    update->maxtime   = -1;
    update->firststep = update->ntimestep;
    update->laststep  = update->firststep + nsteps;
    ctx->condition    = Var("timestep<"+to_string(update->laststep), update->ntimestep<update->laststep);
    ctx->pMpm->output->write(update->ntimestep);
    ctx->breakLoop = false;
    this->addResult(ctx);
  }

private:
  std::atomic_bool              canTerminate_ = false;
  std::vector<std::string>      lines_;
  hh::Graph<1, MPM, float>      *pGraph = nullptr;
};

class ResetGridTask: public hh::AbstractTask<1, HHContext, HHContext> {
public:
  explicit ResetGridTask(): hh::AbstractTask<1, HHContext, HHContext>("ResetGrid", 1, false) {}

  void execute(std::shared_ptr<HHContext> ctx) override {
    if constexpr(LOG_INFO) printf("[%s:%d]\n", __FUNCTION__, __LINE__);
    auto update = ctx->pMpm->update;
    auto modify = ctx->pMpm->modify;

    ctx->ntimestep = update->update_timestep();
    update->method->compute_grid_weight_functions_and_gradients();
    update->method->reset();
    modify->initial_integrate();
    this->addResult(ctx);
  }
};

class ParticlesToGridTask: public hh::AbstractTask<1, HHContext, HHContext> {
public:
  explicit ParticlesToGridTask(): hh::AbstractTask<1, HHContext, HHContext>("P2G", 1, false) {}

  void execute(std::shared_ptr<HHContext> ctx) override {
    if constexpr(LOG_INFO) printf("[%s:%d]\n", __FUNCTION__, __LINE__);
    auto update = ctx->pMpm->update;
    update->method->particles_to_grid();//@MPI MPI_Allreduce 1 element, check for rigid solids
    this->addResult(ctx);
  }
};

class UpdateGridStateTask: public hh::AbstractTask<1, HHContext, HHContext> {
public:
  explicit UpdateGridStateTask(): hh::AbstractTask<1, HHContext, HHContext>("UpdateGridStateTask", 1, false) {}

  void execute(std::shared_ptr<HHContext> ctx) override {
    if constexpr(LOG_INFO) printf("[%s:%d]\n", __FUNCTION__, __LINE__);
    auto update = ctx->pMpm->update;
    auto modify = ctx->pMpm->modify;
    update->method->update_grid_state();
    modify->post_update_grid_state();//@MPI MPI_Allreduce in FixVelocityNodes, Vector3 forceTotal
    this->addResult(ctx);
  }
};

class GridToParticles: public hh::AbstractTask<1, HHContext, HHContext> {
public:
  explicit GridToParticles(): hh::AbstractTask<1, HHContext, HHContext>("G2P", 1, false) {}

  void execute(std::shared_ptr<HHContext> ctx) override {
    if constexpr(LOG_INFO) printf("[%s:%d]\n", __FUNCTION__, __LINE__);
    auto update = ctx->pMpm->update;
    update->method->grid_to_points();
    this->addResult(ctx);
  }
};

class IntermediateTask: public hh::AbstractTask<1, HHContext, HHContext> {
public:
  explicit IntermediateTask(): hh::AbstractTask<1, HHContext, HHContext>("IntermediateTask", 1, false) {}

  void execute(std::shared_ptr<HHContext> ctx) override {
    if constexpr(LOG_INFO) printf("[%s:%d]\n", __FUNCTION__, __LINE__);
    auto update = ctx->pMpm->update;
    auto modify = ctx->pMpm->modify;
    update->method->advance_particles();
    update->method->velocities_to_grid();
    modify->post_velocities_to_grid();
    this->addResult(ctx);
  }
};

class DeformationGradientTask: public hh::AbstractTask<1, HHContext, HHContext> {
public:
  explicit DeformationGradientTask(): hh::AbstractTask<1, HHContext, HHContext>("DeformationGradientTask", 1, false) {}

  void execute(std::shared_ptr<HHContext> ctx) override {
    if constexpr(LOG_INFO) printf("[%s:%d]\n", __FUNCTION__, __LINE__);
    auto update = ctx->pMpm->update;
    update->method->compute_rate_deformation_gradient(true);
    update->method->update_deformation_gradient();
    this->addResult(ctx);
  }
};

class StressComputationTask: public hh::AbstractTask<1, HHContext, HHContext> {
public:
  explicit StressComputationTask(): hh::AbstractTask<1, HHContext, HHContext>("StressComputationTask", 1, false) {}

  void execute(std::shared_ptr<HHContext> ctx) override {
    if constexpr(LOG_INFO) printf("[%s:%d]\n", __FUNCTION__, __LINE__);
    auto update = ctx->pMpm->update;
    update->method->update_stress(true);
    this->addResult(ctx);
  }
};

class ExchangeParticlesTask: public hh::AbstractTask<1, HHContext, HHContext> {
public:
  explicit ExchangeParticlesTask(): hh::AbstractTask<1, HHContext, HHContext>("ExchangeParticlesTask", 1, false) {}

  void execute(std::shared_ptr<HHContext> ctx) override {
    if constexpr(LOG_INFO) printf("[%s:%d]\n", __FUNCTION__, __LINE__);
    auto update = ctx->pMpm->update;
    update->method->exchange_particles();//@MPI: bunch of mpi sends and recvs
    this->addResult(ctx);
  }
};

class UpdateTimeTask: public hh::AbstractTask<1, HHContext, HHContext> {
public:
  explicit UpdateTimeTask(): hh::AbstractTask<1, HHContext, HHContext>("UpdateTimeTask", 1, false) {}

  void execute(std::shared_ptr<HHContext> ctx) override {
    if constexpr(LOG_INFO) printf("[%s:%d]\n", __FUNCTION__, __LINE__);
    auto update = ctx->pMpm->update;
    update->update_time();
    update->method->adjust_dt();//@MPI MPI_Allreduce 1 element, dtCFL_reduced for calculating the next time step (dt)
    this->addResult(ctx);
  }
};

class OutputTask: public hh::AbstractTask<1, HHContext, HHContext> {
public:
  explicit OutputTask(): hh::AbstractTask<1, HHContext, HHContext>("OutputTask", 1, false) {}

  void execute(std::shared_ptr<HHContext> ctx) override {
    if constexpr(LOG_INFO) printf("[%s:%d]\n", __FUNCTION__, __LINE__);
    auto update = ctx->pMpm->update;
    auto output = ctx->pMpm->output;
    auto ntimestep = ctx->ntimestep;
    if((update->maxtime != -1) && (update->atime > update->maxtime)) {
      update->nsteps = ntimestep;
      output->write(ntimestep);
      ctx->breakLoop = true;
      this->addResult(ctx);
      return;
    }

    if(ntimestep == output->next || ntimestep == update->nsteps) {
      output->write(ntimestep);
    }
    this->addResult(ctx);
  }

};

int main(int argc, char **argv) {
  MPI_Init(&argc,&argv);
  auto mpm = std::make_shared<MPM>(argc, argv, MPI_COMM_WORLD);

  {
    auto graph                   = hh::Graph<1, MPM, float>("Karamelo");
    auto inputTask               = std::make_shared<InputTask>(graph);
    auto resetGridTask           = std::make_shared<ResetGridTask>();
    auto particlesToGridTask     = std::make_shared<ParticlesToGridTask>();
    auto updateGridStateTask     = std::make_shared<UpdateGridStateTask>();
    auto gridToParticles         = std::make_shared<GridToParticles>();
    auto intermediateTask        = std::make_shared<IntermediateTask>();
    auto deformationGradientTask = std::make_shared<DeformationGradientTask>();
    auto stressComputationTask   = std::make_shared<StressComputationTask>();
    auto exchangeParticlesTask   = std::make_shared<ExchangeParticlesTask>();
    auto updateTimeTask          = std::make_shared<UpdateTimeTask>();
    auto outputTask              = std::make_shared<OutputTask>();

    graph.input<MPM>(inputTask);
    graph.edge<HHContext>(inputTask, resetGridTask);
    graph.edge<HHContext>(resetGridTask, particlesToGridTask);
    graph.edge<HHContext>(particlesToGridTask, updateGridStateTask);
    graph.edge<HHContext>(updateGridStateTask, gridToParticles);
    graph.edge<HHContext>(gridToParticles, intermediateTask);
    graph.edge<HHContext>(intermediateTask, deformationGradientTask);
    graph.edge<HHContext>(deformationGradientTask, stressComputationTask);
    graph.edge<HHContext>(stressComputationTask, exchangeParticlesTask);
    graph.edge<HHContext>(exchangeParticlesTask, updateTimeTask);
    graph.edge<HHContext>(updateTimeTask, outputTask);
    graph.edge<HHContext>(outputTask, inputTask);
    graph.output<float>(inputTask);

    graph.executeGraph();
    graph.pushData(mpm);
    graph.finishPushingData();
    graph.waitForTermination();
    graph.createDotFile("temp" + std::to_string(mpm->universe->me) + ".final.dot", hh::ColorScheme::EXECUTION, hh::StructureOptions::ALL, hh::InputOptions::GATHERED, hh::DebugOptions::NONE);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  mpm.reset();
  MPI_Finalize();

  return 0;
}