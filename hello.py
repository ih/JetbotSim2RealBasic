# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/00_sim/create_empty.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.sim import SimulationCfg, SimulationContext
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.assets import Articulation
import omni.isaac.lab.sim as sim_utils
from jetbot import JETBOT_CFG 
import pdb

def design_scene():
    print("NUCLEUS DIR" + ISAAC_NUCLEUS_DIR)
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd")
    cfg.func("/World/Objects/Room", cfg, translation=(0.0, 0.0, 0))

    # load the robot
    prim_utils.create_prim("/World/Origin", "Xform", translation=[0.0, 0.0, 0.0])

    jetbot_cfg = JETBOT_CFG.copy()
    jetbot_cfg.prim_path = "/World/Origin/Robot"
    jetbot = Articulation(cfg=jetbot_cfg)
    return {"jetbot": jetbot} 

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    robot = entities["jetbot"]
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * .1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            robot.reset()
            print("[INFO]: Resetting robot state...")
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        robot.set_joint_effort_target(efforts)
        robot.write_data_to_sim()
        # perform step
        sim.step()
        count +=1
        robot.update(sim_dt)

def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Scene Design
    scene_entities = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # pdb.set_trace()
    run_simulator(sim, scene_entities)



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
