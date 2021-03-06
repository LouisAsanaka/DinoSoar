import imageio
import os
import argparse
import time
import numpy as np
from typing import Optional

from dino_env import ChromeDinoEnv
from utils import *
from wrappers import *

import gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.atari_wrappers import *


config = {
    'batch_size': 256,
    'clip_range': linear_schedule(0.1),
    'ent_coef': 0.01,
    'learning_rate': linear_schedule(2.5e-4),
    'n_epochs': 4,
    'n_steps': 128,
    'vf_coef': 0.5
}


def create_env(env_count: int, screen_width: int = 96, screen_height: int = 96, skip: int = 4) -> SubprocVecEnv:
    def env_wrappers(env: gym.Env) -> gym.Env:
        env = SkipWrapper(skip)(env)
        env = WarpFrame(env, width=screen_width, height=screen_height)
        return env

    venv = make_vec_env(ChromeDinoEnv, n_envs=env_count,
        wrapper_class=env_wrappers,
        env_kwargs={
            'chromedriver_path': os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "chromedriver"
            )
        },
        vec_env_cls=SubprocVecEnv
    )
    venv = VecFrameStack(venv, n_stack=4)
    venv = VecTransposeImage(venv)
    return venv


def train(timesteps: int, save_freq: int, eval_freq: int, env_count: int, 
    previous_model: Optional[str]):
    uid: str = random_model_name()

    log(f'Training model "{uid}"...')
    log(f'Creating environment...')
    env: SubprocVecEnv = create_env(env_count)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=f'./checkpoints/{uid}/',
    )

    eval_env: SubprocVecEnv = create_env(env_count=1)
    eval_callback = EvalCallback(eval_env=eval_env, n_eval_episodes=5,
        eval_freq=eval_freq, best_model_save_path='./best_models/', 
        deterministic=True, verbose=2)

    if previous_model is not None: # Load previous model & continue training
        log(f'Found previous model "{previous_model}"! Loading...')
        model = PPO.load(previous_model, env, **config, 
            verbose=2, tensorboard_log='./tb_dinosoar/')
    else: # Start training from scratch
        log(f'Training from scratch...')
        model = PPO(
            'CnnPolicy', env, **config,
            verbose=2, tensorboard_log='./tb_dinosoar/')
    
    log('Started training!')

    start_time: int = time.time()
    model.learn(
        total_timesteps=timesteps, callback=[checkpoint_callback, eval_callback]
    )
    log(f'Done training in {to_hms(time.time() - start_time)}! Saving model "{uid}"...')
    model.save(uid)
    log(f'Running model & saving gifs...')

    env.reset()
    img = model.env.render(mode='rgb_array')
    imageio.imsave(f'dino_{uid}_train.png', np.array(img))

    eval_env.reset()
    img = eval_env.render(mode='rgb_array')
    imageio.imsave(f'dino_{uid}_eval.png', np.array(img))

    log(f'Done saving image!')


def evaluate(model_name: str):
    if model_name is None:
        log('No model name provided!')
        return
    log(f'Evaluating model "{model_name}"...')
    log(f'Creating environment...')
    env: SubprocVecEnv = create_env(env_count=1)

    model = PPO.load(model_name, env, **config,
        verbose=2, tensorboard_log='./tb_dinosoar/')

    images = []

    obs = env.reset()
    dones = np.array([False] * 1)
    img = model.env.render(mode='rgb_array')

    log('Running environment...')
    i = 0
    while not np.all(dones):
        images.append(img)
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)

        img = env.render(mode='rgb_array')
        i += 1

    log('Saving gif...')
    imageio.mimsave(f'dino_{model_name}.gif', [np.array(img) for i, img in enumerate(images)], fps=15)


def record_best_episode(model_name: str, episode_count: int):
    if model_name is None:
        log('No model name provided!')
        return

    log(f'Evaluating model "{model_name}"...')
    log(f'Creating environment...')
    env: SubprocVecEnv = create_env(env_count=1)

    model = PPO.load(model_name, env, **config,
        verbose=2, tensorboard_log='./tb_dinosoar/')

    best_frames = None
    best_net_reward = 0

    all_rewards = []

    log('Running environment...')
    for episode in range(episode_count):
        log(f'Running episode {episode}...')
        i = 0

        images = []
        obs = env.reset()
        log('Waiting for reset...')
        time.sleep(2)
        dones = np.array([False] * 1)
        img = model.env.render(mode='rgb_array')
        net_reward = 0
        while not np.all(dones):
            images.append(img)
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            net_reward += rewards[0]
            
            img = env.render(mode='rgb_array')
            i += 1
        log('Done!')
        all_rewards.append(net_reward)
        if net_reward > best_net_reward:
            log(f'New best score of {net_reward}! (beat previous score of {best_net_reward})')
            best_frames = images
            best_net_reward = net_reward
        else:
            log(f'Score of {net_reward} does not beat the current high score of {best_net_reward}.')

    log(f'Saving best gif with score of {best_net_reward}...')
    imageio.mimsave(f'dino_best_{model_name}.gif', [np.array(img) for i, img in enumerate(best_frames)], fps=15)

    all_rewards = np.array(all_rewards)
    mean = np.mean(all_rewards)
    std = np.std(all_rewards)
    log(f'Stats: mean={mean}, std={std}')


def save_observation_images():
    log(f'Creating environment...')
    env: SubprocVecEnv = create_env(env_count=1)

    images = []

    obs = env.reset()
    dones = np.array([False] * 1)

    log('Running environment...')
    i = 0
    while not np.all(dones):
        images.append(obs[0])
        obs, rewards, dones, info = env.step(np.array([0]))
        i += 1

    log('Saving gif...')
    save_observations(images, filename=f'obs')


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
    train_parser.add_argument('-e', '--evalfreq', default=10000, type=int,
        help='evalute model every evalfreq timesteps')
    train_parser.add_argument('-c', '--count', default=8, type=int,
        help='number of environments to create')
    train_parser.add_argument('-m', '--model', default=None,
        help='path to previous model to load')

    eval_parser = subparsers.add_parser('eval', help='evaluate the agent')
    eval_parser.set_defaults(which='eval')
    eval_parser.add_argument('model', default=None,
        help='path to previous model to load')

    record_parser = subparsers.add_parser('record', help='record the agent\'s best run')
    record_parser.set_defaults(which='record')
    record_parser.add_argument('model', default=None,
        help='path to previous model to load')
    record_parser.add_argument('-e', '--episodes', default=10, type=int,
        help='number of episodes to run')

    save_obs_parser = subparsers.add_parser('obs', help='save observations images from the game')
    save_obs_parser.set_defaults(which='obs')

    args = parser.parse_args()

    if args.which == 'train':
        train(timesteps=args.timesteps, save_freq=args.savefreq, 
              eval_freq=args.evalfreq, env_count=args.count, 
              previous_model=args.model)
    elif args.which == 'eval':
        evaluate(model_name=args.model)
    elif args.which == 'record':
        record_best_episode(model_name=args.model, episode_count=args.episodes)
    elif args.which == 'obs':
        save_observation_images()

    exit()