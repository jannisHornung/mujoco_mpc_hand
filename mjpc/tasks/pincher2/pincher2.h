#ifndef MJPC_TASKS_HAND_PINCH2_HAND_PINCH2_H_
#define MJPC_TASKS_HAND_PINCH2_HAND_PINCH2_H_

#include <string>
#include <memory>
#include <fstream>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {

class Pincher2 : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;

  Pincher2();
  ~Pincher2();

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  mjpc::BaseResidualFn* InternalResidual() override { return &residual_; } // Return type corrected

 private:
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Pincher2* task) : mjpc::BaseResidualFn(task) {}

    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };

  ResidualFn residual_;
  mutable std::ofstream log_file_;  // Mutable to allow modification in const methods
  mutable int timestep_counter_ = 0;
  const int min_timesteps_ = 0; // Mutable to allow incrementing in const methods
  const int max_timesteps_ = 600000;

  void LogResiduals(const double* residual, int size) const; // Keep this method const
};

}  // namespace mjpc

#endif  // MJPC_TASKS_HAND_PINCH2_HAND_PINCH2_H_