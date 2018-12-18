import gym
import gym_env.gym_polyhash.envs.polyhash_env

if __name__ == '__main__':
    env = gym.make('Polyhash-v0')
    """env.reset()
    for i in range(1000):
        env.step(env.action_space.sample())
        if i == 0:
            print(env.action_space)"""
    for i_episode in range(20):
        env.reset()
        for t in range(100):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        print(env.reward)
