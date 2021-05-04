from random import choice
from string import ascii_lowercase
import imageio
import numpy as np
from typing import List


def to_hms(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return '{}:{:0>2}:{:0>2}'.format(h, m, s)


def random_model_name():
    return ''.join(choice(ascii_lowercase) for i in range(6))


def log(message: str):
    print(f'> {message}')


def save_observation(img: np.ndarray, filename: str):
    imageio.imsave(f'{filename}.png', img[0, :, :])


def save_observations(imgs: List[np.ndarray], filename: str):
    # Take out frame stacking
    imgs = [img[0, :, :] for img in imgs]
    imageio.mimsave(f'{filename}.gif', imgs, fps=15)
