from gym_chrome_dino.envs.chrome_dino_env import ChromeDinoEnv
from gym_chrome_dino.utils.wrappers import WarpFrame

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

import imageio
import numpy as np


def warp_frame(env):
        return WarpFrame(env, 160, 80)

def train():
    print("> Creating environment...")
    env: DummyVecEnv = make_vec_env(
        'ChromeDinoNoBrowser-v0', n_envs=4,
        wrapper_class=warp_frame
    )

    unwrapped_game = env.envs[0].game

    # Constant speed training first
    unwrapped_game.set_parameter('config.ACCELERATION', 0)
    unwrapped_game.set_parameter('config.BG_CLOUD_SPEED', 0.0)
    unwrapped_game.set_parameter('config.CLOUD_FREQUENCY', 0.0)
    unwrapped_game.set_parameter('config.INVERT_DISTANCE', 1000000000)
    unwrapped_game.set_parameter('config.MAX_CLOUDS', 0)

    env = VecFrameStack(env, n_stack=4)

    print("> Creating model...")
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_dino_tensorboard/")
    print("> Learning...")
    model.learn(total_timesteps=2000000)
    print("> Done! Saving...")
    model.save("ppo_dino")


def evaluate():
    print("> Creating environment...")
    env: DummyVecEnv = make_vec_env(
        # 'ChromeDinoNoBrowser-v0', n_envs=1,
        'ChromeDino-v0', n_envs=1,
        wrapper_class=warp_frame
    )

    unwrapped_game = env.envs[0].game

    # Constant speed training first
    unwrapped_game.set_parameter('config.ACCELERATION', 0)
    unwrapped_game.set_parameter('config.BG_CLOUD_SPEED', 0.0)
    unwrapped_game.set_parameter('config.CLOUD_FREQUENCY', 0.0)
    unwrapped_game.set_parameter('config.INVERT_DISTANCE', 1000000000)
    unwrapped_game.set_parameter('config.MAX_CLOUDS', 0)

    env = VecFrameStack(env, n_stack=4)

    print("> Loading model")
    model = PPO.load("ppo_dino")

    images = []
    dones = np.array([False])
    obs = env.reset()
    img = env.render(mode='rgb_array')
    while not np.all(dones):
        images.append(img)
        action, _states = model.predict(obs, deterministic=True)
        print(action)
        obs, rewards, dones, info = env.step(action)
        img = env.render(mode='rgb_array')

    imageio.mimsave('dino.gif', [np.array(img) for i, img in enumerate(images)], fps=15)
    env.close()

# Run tensorboard remote: https://stackoverflow.com/a/40413202/11323583
def main():
    train()

if __name__ == "__main__":
    main()