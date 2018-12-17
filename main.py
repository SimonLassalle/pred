import gym
import gym_env.gym_polyhash.envs.polyhash_env

if __name__ == '__main__':
    env = gym.make('Polyhash-v0')
    env.reset()
    for _ in range(1000):
        env.step(env.action_space.sample())
