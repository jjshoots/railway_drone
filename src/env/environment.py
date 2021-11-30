import os
import cv2
import glob
import math
import time
import numpy as np

import matplotlib.pyplot as plt

from env.aviary import *

class Environment():
    """
    Wrapper for Aviary and Drone Classes with domain randomization
    """
    def __init__(self, rails_dir, drone_dir, plants_dir, tex_dir, num_envs, max_steps=math.inf):
        self.max_steps = max_steps
        self.rails_dir = rails_dir
        self.drone_dir = drone_dir
        self.plants_dir = plants_dir
        self.texture_paths = glob.glob(os.path.join(tex_dir, '**', '*.jpg'), recursive=True)

        self.render = num_envs == 1

        self.num_actions = 2

        self.reset()


    def reset(self):
        try:
            self.env.disconnect()
        except:
            pass

        self.step_count = 0

        self.env = Aviary(rails_dir=self.rails_dir, drone_dir=self.drone_dir, plants_dir=self.plants_dir, render=self.render)
        self.env.drone.set_mode(4)
        self.frame_size = self.env.drone.frame_size

        # clear argsv[0] message, I don't know why it works don't ask me why it works
        print ("\033[A                             \033[A")

        self.update_textures()

        # wait for env to stabilize
        for _ in range(10):
            self.env.step()
            self.env.drone.setpoint = np.array([0, 0, 0, 2])

        # track state is dist, angle
        # drone state is xy velocity
        self.track_state = self.env.track_state()
        self.drone_state = self.env.drone.state[-2][:2]


    def get_state(self):
        return self.env.drone.rgbImg, self.drone_state, 0., 0., self.track_state


    def step(self, target):
        """
        step the entire simulation
            input is the railstate as [pos, orn]
            output is tuple of observation(ndarray), reward(scalar), done(int), trackstate([pos, orn])
        """
        reward = 0.
        done = 0.
        while not self.env.step():
            # reward is computed for the previous time step
            reward = -np.linalg.norm(self.track_state)

            # step the env
            self.env.drone.setpoint = self.tgt2set(target * self.env.track_state_norm)
            self.step_count += 1

            # every 240 time steps (1 seconds), change the texture of the floor
            if self.step_count % 240 == 1:
                self.update_textures()

            # get the states
            self.track_state = self.env.track_state()
            self.drone_state = self.env.drone.state[-2][:2]

            # check terminate
            done = 1. if self.step_count >= self.max_steps else 0.
            if np.isnan(np.sum(self.track_state)):
                done = 1.
                self.track_state = np.array([0., 0.])

        # label = self.track_state if np.linalg.norm(self.track_state) > 0.4 else np.zeros_like(self.track_state)
        # label = np.clip(self.track_state, 0., 1.)
        label = self.track_state

        return self.env.drone.rgbImg, self.drone_state, reward, done, label


    def tgt2set(self, track_state: np.ndarray) -> np.ndarray:
        gain = 2.

        c = np.cos(track_state[1])
        s = np.sin(track_state[1])
        rot = (np.array([[c, -s], [s, c]]))

        setpoint = np.matmul(rot, np.array([[-gain * track_state[0]], [6.]])).flatten()

        setpoint = np.array([*setpoint, gain * track_state[1], 2.])

        return setpoint


    def update_textures(self):
        """
        randomly change the texture of the env
        25% chance of the rail being same texture as floor
        25% chance of clutter being same texture as rails
        25% chance of rail, floor, and clutter being same texture
        25% chance of all different
        """

        chance = np.random.randint(4)

        if chance == 0:
            # rail same as floor, clutter diff
            tex_id = self.get_random_texture()
            for rail in self.env.rails:
                rail.change_rail_texture(tex_id)
            self.env.changeVisualShape(self.env.planeId, -1, textureUniqueId=tex_id)

            tex_id = self.get_random_texture()
            self.env.clutter.change_rail_texture(tex_id)
        elif chance == 1:
            # rail same as clutter, floor diff
            tex_id = self.get_random_texture()
            for rail in self.env.rails:
                rail.change_rail_texture(tex_id)
            self.env.clutter.change_rail_texture(tex_id)

            tex_id = self.get_random_texture()
            self.env.changeVisualShape(self.env.planeId, -1, textureUniqueId=tex_id)
        elif chance == 2:
            # all same
            tex_id = self.get_random_texture()
            for rail in self.env.rails:
                rail.change_rail_texture(tex_id)
            self.env.clutter.change_rail_texture(tex_id)
            self.env.changeVisualShape(self.env.planeId, -1, textureUniqueId=tex_id)
        else:
            # all diff
            tex_id = self.get_random_texture()
            for rail in self.env.rails:
                rail.change_rail_texture(tex_id)

            tex_id = self.get_random_texture()
            self.env.clutter.change_rail_texture(tex_id)

            tex_id = self.get_random_texture()
            self.env.changeVisualShape(self.env.planeId, -1, textureUniqueId=tex_id)


    def get_random_texture(self) -> int:
        texture_path = self.texture_paths[np.random.randint(0, len(self.texture_paths) - 1)]
        tex_id = -1
        while tex_id < 0:
            tex_id = self.env.loadTexture(texture_path)
        return tex_id

