import cv2
import math
import time
import numpy as np

import matplotlib.pyplot as plt

from env.aviary import *
from env.texturePack import *

class Environment():
    """
    Wrapper for Aviary and Drone Classes
    """
    def __init__(self, rails_dir, drone_dir, tex_dir, max_steps=2000):
        self.max_steps = max_steps
        self.rails_dir = rails_dir
        self.drone_dir = drone_dir
        self.tex_dir = tex_dir

        self.reset()


    def reset(self):
        try:
            self.env.disconnect()
        except:
            pass

        self.step_count = 0

        self.env = Aviary(rails_dir=self.rails_dir, drone_dir=self.drone_dir, render=False)
        self.tex = TexturePack(self.env, self.tex_dir)
        self.env.drone.set_mode(4)

        self.update_textures()

        for _ in range(100):
            self.env.step()
            self.env.drone.setpoint = np.array([0, 0, 0, 2])

        self.track_state = self.env.track_state()


    def step(self, action):
        """
        step the entire simulation
            input is the railstate as [pos, orn]
            output is tuple of observation(ndarray), reward(scalar), done(int), trackstate([pos, orn])
        """
        # reward is computed for the previous time step
        reward = -np.linalg.norm(action - self.track_state)

        # step the env
        if self.env.step():
            self.env.drone.setpoint = self.env.flight_target(action)
        self.step_count += 1

        # every 480 time steps (2 seconds), change the texture of the floor
        if self.step_count % 480 == 1:
            self.update_textures()

        # get the new track_state
        self.track_state = self.env.track_state()

        # check terminate
        done = 1. if self.step_count >= self.max_steps else 0.
        if np.isnan(np.sum(self.track_state)):
            done = 1.
            reward = -100

        return self.env.drone.rgbImg, reward, done, self.track_state


    def update_textures(self):
        tex_id = self.tex.get_random_texture()
        self.env.changeVisualShape(self.env.planeId, -1, textureUniqueId=tex_id)
        tex_id = self.tex.get_random_texture()
        self.env.change_rail_texture(tex_id)

