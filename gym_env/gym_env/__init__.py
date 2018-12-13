from gym.envs.registration import register

register(
    id='polyhash-v0',
    entry_point='gym_env.envs:PolyhashEnv',
)
