import math
import time
import numpy as np
import numpy.polynomial.polynomial as polynomial

import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

from drone import *
from railObject import *

class Environment(bullet_client.BulletClient):
    def __init__(self, render=True):
        super().__init__(p.GUI if render else p.DIRECT)

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

        # start rail graph
        start_pos = np.array([0, 0, 0])
        start_orn = np.array([0.5*math.pi, 0, 0])
        rail = RailObject(self, start_pos, start_orn, 'tracks_obj/rail_straight.obj')
        self.railIds = np.array([rail.Id])
        self.rail_head = rail.get_end(0)
        self.rail_tail = rail.get_end(1)

        # spawn drone
        self.drone = Drone(self, camera_Hz=self.camera_Hz)
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

        self.drone.update()

        self.stepSimulation()
        self.step_count += 1

        if self.step_count % self.rel_cam_Hz == 0:
            self.handle_rail_bounds()
            self.drone.capture_image()
            return True


    def handle_rail_bounds(self):
        dis2head = np.sum((self.rail_head.base_pos[:2] - self.drone.state[-1][:2]) ** 2) ** 0.5
        dis2tail = np.sum((self.rail_tail.base_pos[:2] - self.drone.state[-1][:2]) ** 2) ** 0.5

        # delete the head if it's too far and get the new one
        if dis2head > 20:
            deleted, self.rail_head = self.rail_head.delete(0)
            self.railIds = [id for id in self.railIds if id not in deleted]

        # create new tail if it's too near
        if dis2tail < 20:
            self.rail_tail.add_child('tracks_obj/rail_straight.obj')
            self.rail_tail = self.rail_tail.get_end(1)
            self.railIds = np.append(self.railIds, self.rail_tail.Id)


    def get_flight_target(self):
        railImg = np.isin(self.drone.segImg, self.railIds)

        if np.sum(railImg) > 1:
            angles = self.drone.a_array[railImg.flatten()]

            y = self.drone.state[3][-1] / np.cos(angles[:, 1]) * np.cos(angles[:, 0])
            x = self.drone.state[3][-1] / np.cos(angles[:, 1]) * np.sin(angles[:, 0])

            poly = polynomial.Polynomial.fit(y, x, 1).convert(domain=(-1, 1))
            location = polynomial.polyval(1., [*poly])
            gradient = math.atan(polynomial.polyval(1., [*poly.deriv()]))

            c = np.cos(gradient)
            s = np.sin(gradient)
            rot = (np.array([[c, -s], [s, c]]))

            vel = np.matmul(rot, np.array([[-location], [3.]])).flatten()

            target = np.array([*vel, gradient, 2.])
        else:
            target = np.array([0, 0, 0, 2.])

        return target
