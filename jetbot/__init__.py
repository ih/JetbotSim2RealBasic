"""
Jetbot balancing environment.
"""

from . import agents
import gymnasium as gym
from .jetbotenv import JetbotEnvCfg 
print("===============================")
print("Registering Jetbot Environtment")
print("===============================")

gym.register(
    id="Isaac-Jetbot-Direct-v0",
    entry_point="jetbot.jetbotenv:JetbotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": JetbotEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml"
    },
)

