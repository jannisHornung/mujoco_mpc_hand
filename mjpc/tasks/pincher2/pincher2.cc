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
  double* target = SensorByName(model, data, "target");
  mju_sub(residual + counter, index_tip, target, 3);
  counter += 3;
  mju_sub(residual + counter, thumb_tip, target, 3);
  counter += 3;

  CheckSensorDim(model, counter);

  const Pincher2* task = dynamic_cast<const Pincher2*>(this->task_);
  if (task) {
    task->LogResiduals(residual, counter); // Calling the const LogResiduals
  }
}

}  // namespace mjpc