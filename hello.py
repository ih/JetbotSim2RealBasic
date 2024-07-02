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
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.sim import SimulationCfg, SimulationContext
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.assets import Articulation
import omni.isaac.lab.sim as sim_utils
from jetbot import JETBOT_CFG 

def design_scene():
    print("NUCLEUS DIR" + ISAAC_NUCLEUS_DIR)
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd")
    cfg.func("/World/Objects/Room", cfg, translation=(0.0, 0.0, 0))

    # load the robot
    prim_utils.create_prim("/World/Origin", "Xform", translation=[0.0, 0.0, 0.0])

    jetbot_cfg = JETBOT_CFG.copy()
    jetbot_cfg.prim_path = "/World/Origin/Robot"
    jetbot = Articulation(cfg=jetbot_cfg)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Scene Design
    design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
