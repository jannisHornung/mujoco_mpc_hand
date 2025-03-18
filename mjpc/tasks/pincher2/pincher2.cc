#include "mjpc/tasks/pincher2/pincher2.h"

#include <string>
#include <fstream>
#include <ctime> // Include the header for date and time
#include <sstream> // Include the header for string streams

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {

std::string Pincher2::XmlPath() const {
  return GetModelPath("pincher2/task.xml");
}
std::string Pincher2::Name() const { return "Pincher2"; }

Pincher2::Pincher2() : residual_(this) {
  // Get current time and date
  std::time_t t = std::time(nullptr);
  std::tm* now = std::localtime(&t);

  // Format the filename with the current date and time
  std::stringstream filename;
  filename << "/home/robotlab/Documents/GitHub/mujoco_mpc_hand/Measurements/residual_log_pincher2_"
           << (now->tm_year + 1900) << '-' 
           << (now->tm_mon + 1) << '-'
           << now->tm_mday << '_'
           << now->tm_hour << '-'
           << now->tm_min << '-'
           << now->tm_sec << ".txt";

  // Open the file
  log_file_.open(filename.str());
}

Pincher2::~Pincher2() {
  if (log_file_.is_open()) {
    log_file_.close();
  }
}

void Pincher2::LogResiduals(const double* residual, int size) const {
  if (log_file_.is_open() && timestep_counter_ > min_timesteps_ && timestep_counter_ < max_timesteps_) {
    for (int i = 0; i < size; ++i) {
      log_file_ << residual[i] << " ";
    }
    log_file_ << std::endl;
  }
  timestep_counter_++;  // Increment the mutable counter
}

void Pincher2::ResidualFn::Residual(const mjModel* model,
                                    const mjData* data,
                                    double* residual) const {
  int counter = 0;

  mju_copy(residual, data->ctrl, model->nu);
  counter += model->nu;

  double* thumb_tip = mjpc::SensorByName(model, data, "thumb_pos");
  double* index_tip = mjpc::SensorByName(model, data, "index_pos");
  double* target = SensorByName(model, data, "target_2");
  mju_sub(residual + counter, index_tip, target, 3);
  counter += 3;
  mju_sub(residual + counter, thumb_tip, target, 3);
  counter += 3;

  // Angle velocities
  double* joint1_v = mjpc::SensorByName(model, data, "ZF_MCP_to_DAU_CMC_angle_v");
  double* joint2_v = mjpc::SensorByName(model, data, "DAU_CMC_to_DAU_MCP_angle_v");
  double* joint3_v = mjpc::SensorByName(model, data, "DAU_MCP_to_DAU_PIP_angle_v");
  double* joint4_v = mjpc::SensorByName(model, data, "DAU_PIP_to_DAU_DIP_angle_v");
  double* joint5_v = mjpc::SensorByName(model, data, "ZF_MCP_to_ZF_PIP2_angle_v");
  double* joint6_v = mjpc::SensorByName(model, data, "ZF_PIP2_to_ZF_PIP1_angle_v");
  double* joint7_v = mjpc::SensorByName(model, data, "ZF_PIP1_to_ZF_DIP_angle_v");

  double angle_velo = *joint1_v+*joint2_v+*joint3_v+*joint4_v+*joint5_v+*joint6_v+*joint7_v;
  residual[counter++] = angle_velo;

  // get orientation
  double* thumb_tip_2 = mjpc::SensorByName(model, data, "thumb_pos2");
  double* index_tip_2 = mjpc::SensorByName(model, data, "index_pos2");

  double thumb_orientation[3], index_orientation[3];
  mju_sub(thumb_orientation, thumb_tip_2, thumb_tip, 3);
  mju_sub(index_orientation, index_tip_2, index_tip, 3);

  // Normalize the orientation vectors
  mju_normalize(thumb_orientation, 3);
  mju_normalize(index_orientation, 3);
  
  // Compute direction vectors from fingertips to the target
  double thumb_to_target[3], index_to_target[3];
  mju_sub(thumb_to_target, thumb_tip, target, 3);
  mju_sub(index_to_target, index_tip, target, 3);

  // Normalize direction vectors
  mju_normalize(thumb_to_target, 3);
  mju_normalize(index_to_target, 3);

  // Calculate dot products to determine alignment
  double thumb_alignment = mju_dot3(thumb_orientation, thumb_to_target);
  double index_alignment = mju_dot3(index_orientation, index_to_target);

  // Residuals should approach 1 when aligned perfectly (dot product = 1)
  residual[counter++] = 1.0 - thumb_alignment;
  residual[counter++] = 1.0 - index_alignment;


  CheckSensorDim(model, counter);

  const Pincher2* task = dynamic_cast<const Pincher2*>(this->task_);
  if (task) {
    task->LogResiduals(residual, counter); // Calling the const LogResiduals
  }
}

}  // namespace mjpc