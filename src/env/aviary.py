import math
import time
import numpy as np
import numpy.polynomial.polynomial as polynomial

import matplotlib.pyplot as plt

import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

from env.drone import *
from env.railObject import *

class Aviary(bullet_client.BulletClient):
    def __init__(self, rails_dir, drone_dir, render=True):
        super().__init__(p.GUI if render else p.DIRECT)

        self.rails_dir = rails_dir
        self.drone_dir = drone_dir

        # default physics looprate is 240 Hz
        self.period = 1. / 240.
        self.camera_Hz = 24
        self.rel_cam_Hz = int(240 / self.camera_Hz)
        self.now = time.time()

        self.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.render = render
        self.reset()


    def reset(self):
        self.resetSimulation()
        self.setGravity(0, 0, -9.81)
        self.step_count = 0

        """ CONSTRUCT THE WORLD """
        self.planeId = self.loadURDF(
            "plane.urdf",
            useFixedBase=True
        )

        # start rails, the first rail in the list is the main rail to follow
        start_pos = np.array([0, 0, 0])
        start_orn = np.array([0.5*math.pi, 0, 0])
        self.rails = [RailObject(self, start_pos, start_orn, self.rails_dir)]

        # spawn drone
        self.drone = Drone(self, drone_dir=self.drone_dir, camera_Hz=self.camera_Hz)
        self.drone.reset()


    def step(self):
        """
        Steps the environment and also does the camera capture

        Output:
            True if camera has been used
        """
        if self.render:
            elapsed = time.time() - self.now
            time.sleep(max(self.period - elapsed, 0.))
            self.now = time.time()

            # print(f'RTF: {self.period / (elapsed + 1e-6)}')

        self.drone.update()

        self.stepSimulation()
        self.step_count += 1

        if self.step_count % self.rel_cam_Hz == 0:
            for rail in self.rails:
                rail.handle_rail_bounds(self.drone.state[-1][:2])
            self.drone.capture_image()
            return True


    def track_state(self) -> np.ndarray:
        railImg = np.isin(self.drone.segImg, self.rails[0].Ids)

        # ensure that there is a sufficient number of points to run polyfit
        if np.sum(railImg) > self.drone.seg_size[1]:
            proj = self.drone.inv_proj[railImg.flatten()] * self.drone.state[-1][-1]

            poly = polynomial.Polynomial.fit(proj[:, 1], proj[:, 0], 2).convert(domain=(-1, 1))
            pos = polynomial.polyval(1., [*poly])
            orn = math.atan(polynomial.polyval(1., [*poly.deriv()]))

            # plt.scatter(proj[:, 1], proj[:, 0])
            # plt.plot(*poly.linspace(n=100, domain=(0, np.max(proj[:, 1]))), 'y')
            # plt.show()
            # exit()

            return np.array([pos, orn])

        else:
            return np.array([np.NaN, np.NaN])



    def flight_target(self, track_state: np.ndarray) -> np.ndarray:
        c = np.cos(track_state[1])
        s = np.sin(track_state[1])
        rot = (np.array([[c, -s], [s, c]]))

        vel = np.matmul(rot, np.array([[-2 * track_state[0]], [6.]])).flatten()

        target = np.array([*vel, 2 * track_state[1], 2.])

        return target
