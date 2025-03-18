import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np
import pathlib
import sys
import os
  
# set current directory: mujoco_mpc/python/mujoco_mpc
from mujoco_mpc import agent as agent_lib
#%%
model_path = (
    pathlib.Path(__file__).parent.parent.parent
    / "../build/mjpc/tasks/pincher2/task.xml"
)
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

width, height = 480, 480  # Resolution of the video
frames = []  # Store frames for the video

# Create MuJoCo renderer context for capturing images
renderer = mujoco.Renderer(model, width, height)
#%%
agent = agent_lib.Agent(task_id="Pincher2", model=model)
print("Cost weights:", agent.get_cost_weights())
print("Parameters:", agent.get_task_parameters())

# rollout horizon
T = 1000
frames = [None] * (T-1)

# trajectories
qpos = np.zeros((model.nq, T))
qvel = np.zeros((model.nv, T))
ctrl = np.zeros((model.nu, T - 1))
time = np.zeros(T)

# costs
cost_total = np.zeros(T - 1)
cost_terms = np.zeros((len(agent.get_cost_term_values()), T - 1))

# rollout
mujoco.mj_resetData(model, data)

# cache initial state
time[0] = data.time

# First, get the ID of the "target" body using mj_name2id
#target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")

# simulate
for t in range(T - 1):
  if t % 100 == 0:
    print("t = ", t)

  # set planner state
  agent.set_state(
      time=data.time,
      qpos=data.qpos,
      qvel=data.qvel,
      act=data.act,
      mocap_pos=data.mocap_pos,
      mocap_quat=data.mocap_quat,
      userdata=data.userdata,
  )

  # run planner for num_steps
  num_steps = 10
  for _ in range(num_steps):
    agent.planner_step()

  # Move the target every 5 steps in the negative z-direction
  if t % 5 == 0 and model.nmocap >0 and t < 100:
    data.mocap_pos[0, 2] -= 0.00005  # Move in the negative z direction (adjust 0.05 to desired amount)

  # set ctrl from agent policy
  data.ctrl = agent.get_action()
  ctrl[:, t] = data.ctrl

  # get costs
  cost_total[t] = agent.get_total_cost()
  for i, c in enumerate(agent.get_cost_term_values().items()):
    cost_terms[i, t] = c[1]

  # step
  mujoco.mj_step(model, data)

  # cache
  time[t + 1] = data.time
  renderer.update_scene(data)
  pixels = renderer.render()
  frames[t] = pixels 

# reset
agent.reset()

renderer.close()

#%%
fps = round(T / (time[-1] - time[0]))

import datetime

# Get current date and time
now = datetime.datetime.now()

# Format the date and time as YYYY-MM-DD_HH-MM-SS
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

save_directory = 'home/jannishornung/mujoco_mpc_hand/mujoco_mpc/pinching_for_freedom/videos'

# Construct the file name with the timestamp
output_file = os.path.join(save_directory,f"simulation_{timestamp}.mp4")

media.write_video(output_file, frames, fps=fps)  # Adjust fps as needed

print(f"Video saved as {output_file}")

#%%
# plot control
fig1, ax = plt.subplots()

# Loop through each control and plot it
for i in range(ctrl.shape[0]):
    ax.plot(time[:-1], ctrl[i, :], label=f"Control {i+1}")

# Adding labels and title for control plot
ax.set_xlabel("Time (s)")
ax.set_ylabel("Control")
ax.set_title("Control Variables Over Time")
ax.legend()


# Plot costs
fig2, ax2 = plt.subplots()  # Create a new figure for costs plot

# Loop through each cost term and plot it
for i, c in enumerate(agent.get_cost_term_values().items()):
    ax2.plot(time[:-1], cost_terms[i, :], label=c[0])

# Plot the total cost (weighted)
ax2.plot(time[:-1], cost_total, label="Total (weighted)", color="black")

# Adding labels and title for cost plot
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Costs")
ax2.set_title("Costs Over Time")
ax2.legend()

# Show the plot for costs
plt.show()
# %%
