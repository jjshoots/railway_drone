import cv2
import numpy as np

import matplotlib.pyplot as plt

from env.environment import *

def main():
    grid_size = 4
    num_envs = grid_size ** 2
    envs = [
        Environment(
            rails_dir='models/rails/',
            drone_dir='models/vehicles/',
            tex_dir='models/textures/',
            num_envs=num_envs,
            max_steps=700      # experimentally, 700 steps is sufficient for convergence
            )
        for _ in range(num_envs)
        ]

    track_state = np.zeros((num_envs, 2))
    stack_obs = [None] * num_envs

    cv2.namedWindow('display', cv2.WINDOW_NORMAL)

    while True:
        for i, env in enumerate(envs):
            obs, rew, done, info = env.step(track_state[i])

            if done:
                env.reset()

            stack_obs[i] = obs
            track_state[i] = info

        img = []
        for i in range(grid_size):
            img.append(np.concatenate(stack_obs[i*grid_size:i*grid_size+grid_size], axis=1))
        img = np.concatenate(img, axis=0)

        cv2.imshow('display', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
