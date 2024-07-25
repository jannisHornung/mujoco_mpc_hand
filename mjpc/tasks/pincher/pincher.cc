#include "mjpc/tasks/pincher/pincher.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {
std::string Pincher::XmlPath() const {
  return GetModelPath("pincher/task.xml");
}
std::string Pincher::Name() const { return "Pincher"; }
// ----------------- Residuals for swimmer task ----------------
//   Number of residuals: 11
//     Residual (0-7): control
//     Residual (8-10): XYz displacement between index and thumb
// -------------------------------------------------------------
void Pincher::ResidualFn::Residual(const mjModel* model,
                                     const mjData* data,
                                     double* residual) const {
  // initialize counter
  int counter = 0;

  // "Control"
  mju_copy(residual, data->ctrl, model->nu);
  counter += model->nu; // increment counter

  // "Distance between thumb_tip and index_tip"
  double* thumb_tip = mjpc::SensorByName(model, data, "thumb_pos");
  double* index_tip = mjpc::SensorByName(model, data, "index_pos");
  mju_sub(residual + counter, thumb_tip, index_tip, 3); // Assuming 3D positions
  counter += 3;

  // test residual counter (recommended, optional)
  CheckSensorDim(model, counter);
}
}