#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# Licensed under the MIT License - https://opensource.org/licenses/MIT

import cv2
from PIL import Image
import numpy as np

import gym
from gym import spaces

from gym_chrome_dino.utils.atari_wrappers import FrameStack
from gym_chrome_dino.utils.helpers import Timer

cv2.ocl.setUseOpenCL(False)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width, height):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[:150, :500]
        image = cv2.resize(image, (self.width, self.height))

        # Debug: Save image
        # im = Image.fromarray(image)
        # im.save("lol.png")
        return np.expand_dims(image, axis=2) # add third dimension

class TimerEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.timer = Timer()
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.timer.tick()
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['timedelta'] = self.timer.tick()
        return obs, reward, done, info

def make_dino(env, timer=True, frame_stack=True):
    env = WarpFrame(env, 160, 80)
    if timer:
        env = TimerEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env