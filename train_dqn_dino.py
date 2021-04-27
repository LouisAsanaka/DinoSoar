from dinoenv.chrome_dino_env import ChromeDinoEnv
from dinoenv.wrappers import WarpFrame

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

import imageio
import numpy as np
import sys
from random import choice
from string import ascii_lowercase
from typing import Callable


def random_model_name():
    return ''.join(choice(ascii_lowercase) for i in range(6))


def warp_frame(env):
    return WarpFrame(env, 80, 80)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def train(timesteps: int, save_freq: int, prev_model: str = None):
    print("> Creating environment...")
    vec_env: DummyVecEnv = make_vec_env(
        ChromeDinoEnv, n_envs=1, wrapper_class=warp_frame,
        env_kwargs={'render': False, 'accelerate': False, 'autoscale': False},
        vec_env_cls=DummyVecEnv
    )
    vec_env.env_method(method_name='set_default_parameters')

    env = VecFrameStack(vec_env, n_stack=4)

    model = None
    if prev_model is not None:
        print("> Loading previous model...")
        model = DQN.load(prev_model, env, learning_rate=3e-4, 
            verbose=1, tensorboard_log="./dqn_dino_tensorboard/")
    else:
        print("> Creating model...")
        model = DQN("CnnPolicy", env, learning_rate=3e-4, 
            verbose=1, tensorboard_log="./dqn_dino_tensorboard/")
    
    model_name = random_model_name()
    save_path = f"dqn_dino_{model_name}"
    print(f"> Model name: {model_name}")
    print("> Learning...")
    try:
        # Save a checkpoint every few steps
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq, save_path=f"./checkpoints/{model_name}", name_prefix=model_name)

        model.learn(total_timesteps=timesteps, callback=checkpoint_callback)
        print(f"> Done! Saving as {save_path}.zip")
        model.save(save_path)
    except KeyboardInterrupt:
        print("> Quitting prematurely...")
        print(f"> Saving as {save_path}.zip")
        model.save(save_path)
        sys.exit(0)


def evaluate(model_file):
    print("> Creating environment...")
    env: DummyVecEnv = make_vec_env(
        ChromeDinoEnv, n_envs=1, wrapper_class=warp_frame,
        env_kwargs={'render': True, 'accelerate': False, 'autoscale': False}
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
    model = DQN.load(model_file)

    images = []
    dones = np.array([False])
    obs = env.reset()
    img = env.render(mode='rgb_array')
    reward_total = 0.0
    while not np.all(dones):
        images.append(img)
        action, _states = model.predict(obs, deterministic=True)
        print(action)
        obs, rewards, dones, info = env.step(action)
        reward_total += rewards[0]
        img = env.render(mode='rgb_array')
    print(f"Reward: {reward_total}")
    imageio.mimsave('dino.gif', [np.array(img) for i, img in enumerate(images)], fps=15)
    env.close()

def actual_eval(model_file):
    print("> Creating environment...")
    env: DummyVecEnv = make_vec_env(
        ChromeDinoEnv, n_envs=1, wrapper_class=warp_frame,
        env_kwargs={'render': False, 'accelerate': False, 'autoscale': False}
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
    model = DQN.load(model_file)

    print("> Evaluating...")
    mean, std = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True, render=False, return_episode_rewards=False, warn=True)
    print(f"> Mean: {mean} | Std: {std}")

# Run tensorboard remote: https://stackoverflow.com/a/40413202/11323583
def main():
    train(timesteps=6000000, save_freq=60000)#, prev_model='nwjhlr_800000_steps.zip')
    # evaluate('timqkc_1800000_steps.zip')
    # actual_eval('ppo_dino_xbfgep.zip')

if __name__ == "__main__":
    main()