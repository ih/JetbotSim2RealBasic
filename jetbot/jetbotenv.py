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
import gymnasium as gym
import numpy as np
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
    room_cfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/room", spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd"))
    
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
    action_scale = 100.0

    # Scene
    scene: InteractiveSceneCfg = JetbotSceneCfg(num_envs=1, env_spacing=15.0)

    num_channels = 3
    num_observations = num_channels * scene.camera.height * scene.camera.width

class JetbotEnv(DirectRLEnv):
    cfg: JetbotEnvCfg

    def __init__(self, cfg: JetbotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["jetbot"]
        self.robot_camera = self.scene["camera"]
        self.action_scale = self.cfg.action_scale

    def _configure_gym_env_spaces(self):
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations

        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.scene.camera.height, self.cfg.scene.camera.width, self.cfg.num_channels),
        )
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_actions,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

        # RL specifics
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.sim.device)


    def _get_rewards(self) -> torch.Tensor:
        return torch.ones(self.num_envs)
    
    def _get_observations(self) -> dict:
        observations =  self.robot_camera.data.output["rgb"].clone()
        return {"policy": observations}


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return (torch.full((self.num_envs,) , False, device='cuda'), time_out)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

