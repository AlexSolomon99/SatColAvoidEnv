from gymnasium.envs.registration import register
from gym_satellite_ca.env_config_default import get_env_default_kwargs

default_kwargs = get_env_default_kwargs()

register(
     id="CollisionAvoidanceEnv-v0",
     entry_point="gym_satellite_ca.envs:CollisionAvoidanceEnv",
     max_episode_steps=10000,
     kwargs=default_kwargs,
)
