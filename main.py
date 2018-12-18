import gym
import gym_env.gym_polyhash.envs.polyhash_env

if __name__ == '__main__':
    env = gym.make('Polyhash-v0')
    max_reward = 0
    for i_episode in range(20):
        env.reset()
        for t in range(100):
            action = env.action_space.sample()
            print('action ' + str(action))
            print('reward ' + str(env.reward))
            if env.reward > max_reward:
                max_reward = env.reward
            env.render()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    print('max reward = ' + str(max_reward))
