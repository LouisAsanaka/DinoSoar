from random import choice
from string import ascii_lowercase


def to_hms(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return '{}:{:0>2}:{:0>2}'.format(h, m, s)


def random_model_name():
    return ''.join(choice(ascii_lowercase) for i in range(6))


def log(message: str):
    print(f'> {message}')
