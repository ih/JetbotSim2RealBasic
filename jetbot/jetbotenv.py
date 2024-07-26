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


# import argparse

# from omni.isaac.lab.app import AppLauncher

# # create argparser
# parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args_cli = parser.parse_args()
# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

"""Rest everything follows."""
from collections.abc import Sequence
import torch
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.sim import SimulationCfg, SimulationContext
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets import ArticulationCfg, Articulation, AssetBaseCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sensors import CameraCfg
from .jetbot import JETBOT_CFG 
import pdb


@configclass
class JetbotSceneCfg(InteractiveSceneCfg):
    room_cfg = AssetBaseCfg(prim_path="/World/room", spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd"))
    
    jetbot: ArticulationCfg = JETBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis/rgb_camera/jetbot_camera",
        spawn=None,
        height=224,
        width=224,
        update_period=.1
    )

@configclass 
class JetbotEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 5.0
    num_actions = 2
    num_observations = 1 
    action_scale = 100.0

    # Scene
    scene: InteractiveSceneCfg = JetbotSceneCfg(num_envs=1, env_spacing=2.0)

class JetbotEnv(DirectRLEnv):
    cfg: JetbotEnvCfg

    def __init__(self, cfg: JetbotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["jetbot"]
        self.action_scale = self.cfg.action_scale

    def _get_rewards(self) -> torch.Tensor:
        return torch.tensor([1])
    
    def _get_observations(self) -> dict:
        return {"policy": torch.tensor([1])}

        obs = self.scene
        observations = {"policy": obs}
        return observations

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return (torch.tensor([False, time_out]))

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)





# def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
#     robot = scene["jetbot"]
#     sim_dt = sim.get_physics_dt()
#     count = 0
#     # Simulate physics
#     while simulation_app.is_running():
#         if count % 500 == 0:
#             count = 0
#             joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
#             joint_pos += torch.rand_like(joint_pos) * .1
#             joint_vel = 5
#             robot.write_joint_state_to_sim(joint_pos, joint_vel)

#             scene.reset()
#             print("[INFO]: Resetting robot state...")
#         # efforts = torch.randn_like(robot.data.joint_pos) * 5.0
#         # efforts = robot.data.joint_vel + 10 
#         # print(efforts)

#         # robot.set_joint_effort_target(efforts)
#         robot.set_joint_velocity_target(torch.tensor([[100.0, 100.0]]))
#         scene.write_data_to_sim()
#         # perform step
#         sim.step()
#         count +=1
#         scene.update(sim_dt)
#         # print("-------------------------------")
#         # print(scene["camera"])
#         # print("Received shape of rgb   image: ", scene["camera"].data.output["rgb"].shape)
#         # print(scene["camera"].data.output["rgb"])
#         # print("-------------------------------")

# def main():
#     """Main function."""
#     # Initialize the simulation context
#     sim_cfg = SimulationCfg(dt=0.01)
#     sim = SimulationContext(sim_cfg)
#     # Set main camera
#     sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

#     # Scene Design
#     scene_cfg = JetbotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
#     scene = InteractiveScene(scene_cfg)    # Play the simulator
#     sim.reset()
#     # Now we are ready!
#     print("[INFO]: Setup complete...")
#     # pdb.set_trace()
#     run_simulator(sim, scene)



# if __name__ == "__main__":
#     # run the main function
#     main()
#     # close sim app
#     simulation_app.close()
