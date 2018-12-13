import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Polyhash-v0',
    entry_point='gym_polyhash.envs:PolyhashEnv',
)
