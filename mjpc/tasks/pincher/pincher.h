#ifndef MJPC_TASKS_HAND_PINCH_HAND_PINCH_H_
#define MJPC_TASKS_HAND_PINCH_HAND_PINCH_H_

#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
class Pincher : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Pincher* task) : mjpc::BaseResidualFn(task) {}

    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };

  Pincher() : residual_(this) {}

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};
}  // namespace mjpc

#endif  // MJPC_TASKS_HAND_PINCH_HAND_PINCH_H_
