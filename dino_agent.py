import imageio
import os
import argparse
import time
import numpy as np
from typing import Optional

from dino_env import ChromeDinoEnv
from utils import *

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback


def create_env(env_count: int) -> SubprocVecEnv:
    return make_vec_env(ChromeDinoEnv, n_envs=env_count,
        env_kwargs={
            'screen_width': 96, 
            'screen_height': 96, 
            'chromedriver_path': os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "chromedriver"
            )
        },
        vec_env_cls=SubprocVecEnv
    )


def train(timesteps: int, save_freq: int, env_count: int, 
    previous_model: Optional[str]):
    uid: str = random_model_name()

    log(f'Training model "{uid}"...')
    log(f'Creating environment...')
    env: SubprocVecEnv = create_env(env_count)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=f'./checkpoints/{uid}/',
    )

    if previous_model is not None: # Load previous model & continue training
        log(f'Found previous model "{previous_model}"! Loading...')
        model = PPO.load(previous_model, env, 
            verbose=2, tensorboard_log='./tb_dinosoar/')
    else: # Start training from scratch
        log(f'Training from scratch...')
        model = PPO(
            'CnnPolicy', env,
            verbose=2, tensorboard_log='./tb_dinosoar/')
    
    log('Started training!')

    start_time: int = time.time()
    model.learn(
        total_timesteps=timesteps, callback=[checkpoint_callback]
    )
    log(f'Done training in {to_hms(time.time() - start_time)}! Saving model "{uid}"...')
    model.save(uid)


def evaluate(model: str):
    log(f'Evaluating model "{model}"...')
    log(f'Creating environment...')
    env: SubprocVecEnv = create_env(env_count=1)
    images = []

    obs = env.reset()
    dones = np.array([False])
    img = model.env.render(mode='rgb_array')

    log('Running environment...')
    i = 0
    while not np.all(dones):
        images.append(img)
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        img = env.render(mode='rgb_array')
        i += 1

    log('Saving image...')
    imageio.mimsave(f'dino_{model}.gif', [np.array(img) for i, img in enumerate(images)], fps=15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train / evaluate a RL agent for the Chrome T-Rex run!')
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train', help='train the agent')
    train_parser.set_defaults(which='train')

    # Training arguments
    train_parser.add_argument('-t', '--timesteps', default=2000000, type=int,
        help='number of timesteps to train the agent for')
    train_parser.add_argument('-s', '--savefreq', default=100000, type=int,
        help='save the model every savefreq timesteps')
    train_parser.add_argument('-c', '--count', default=8, type=int,
        help='number of environments to create')
    train_parser.add_argument('-m', '--model', default=None,
        help='path to previous model to load')

    eval_parser = subparsers.add_parser('eval', help='evaluate the agent')
    eval_parser.set_defaults(which='eval')
    eval_parser.add_argument('model', default=None,
        help='path to previous model to load')
    args = parser.parse_args()

    if args.which == 'train':
        train(timesteps=args.timesteps, save_freq=args.savefreq, 
            env_count=args.count, previous_model=args.model)
    elif args.which == 'eval':
        evaluate(model=args.model)

    exit()