from gym.envs.registration import register

register(
    id='Polyhash-v0',
    entry_point='gym_env.gym_polyhash.envs.polyhash_env:PolyhashEnv',
)
