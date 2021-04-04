import gym
import gym_chrome_dino
from gym_chrome_dino.envs.chrome_dino_env import ChromeDinoEnv
from gym_chrome_dino.utils.wrappers import make_dino

env: ChromeDinoEnv = gym.make('ChromeDino-v0')
env = make_dino(env, timer=False, frame_stack=True)
# Constant speed training first
env.unwrapped.set_acceleration(False)
env.reset()

done = False
while not done:
    env.render(mode='rgb_array')
    observation, reward, done, info = env.step(1)
    print(observation)
env.close()
