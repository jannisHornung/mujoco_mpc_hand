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
  double* target = SensorByName(model, data, "target");
  mju_sub(residual + counter, index_tip, target, 3); // Assuming 3D positions
  counter += 3;
  mju_sub(residual + counter, thumb_tip, target, 3); // Assuming 3D positions
  counter += 3;
  // Angles 
  //double* TDP = mjpc::SensorByName(model, data, "TDP_angle_s");
  //double* TIP = mjpc::SensorByName(model, data, "TIP_angle_s");
  //double* TPP = mjpc::SensorByName(model, data, "TPP_angle_s");
  //double* IDP = mjpc::SensorByName(model, data, "IDP_angle_s");
  //double* IIP = mjpc::SensorByName(model, data, "IIP_angle_s");
  //double* IPP = mjpc::SensorByName(model, data, "IPP_angle_s");

  // Angle velocities
  double* TDP_v = mjpc::SensorByName(model, data, "TDP_angle_v");
  double* TIP_v = mjpc::SensorByName(model, data, "TIP_angle_v");
  double* TPP_v = mjpc::SensorByName(model, data, "TPP_angle_v");
  double* IDP_v = mjpc::SensorByName(model, data, "IDP_angle_v");
  double* IIP_v = mjpc::SensorByName(model, data, "IIP_angle_v");
  double* IPP_v = mjpc::SensorByName(model, data, "IPP_angle_v");

  double angle_velo = *TDP_v+*TIP_v+*TPP_v+*IDP_v+*IIP_v+*IPP_v;
  residual[counter++] = angle_velo;
  
 
  //residual[counter++] = *TDP;
  //residual[counter++] = *TIP;
  //residual[counter++] = *TPP;
  //residual[counter++] = *IDP;
  //residual[counter++] = *IIP;
  //residual[counter++] = *IPP;
  // test residual counter (recommended, optional)
  CheckSensorDim(model, counter);
}
}