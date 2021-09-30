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
    def __init__(self, rails_dir, drone_dir, plants_dir, render=True):
        super().__init__(p.GUI if render else p.DIRECT)

        self.rails_dir = rails_dir
        self.drone_dir = drone_dir
        self.plants_dir = plants_dir
        self.initialize_common_meshes()

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
            useFixedBase=True,
            globalScaling=np.random.rand() * 20. + 1.
        )

        # spawn drone
        self.drone = Drone(self, drone_dir=self.drone_dir, camera_Hz=self.camera_Hz)
        self.drone.reset()

        # start rails, the first rail in the list is the main rail to follow
        self.rails = []
        start_pos = np.array([0, 0, 0])
        start_orn = np.array([0.5*math.pi, 0, 0])
        self.rails.append(RailObject(self, start_pos, start_orn, self.rail_mesh))

        # bootstrap off the RailObject to spawn clutter
        self.clutter = RailObject(self, start_pos, start_orn, self.clutter_mesh)


    def initialize_common_meshes(self):
        # rail meshes
        self.rail_mesh = np.ones(3) * -1
        self.rail_mesh[0] = obj_visual(self, self.rails_dir + 'rail_straight.obj')
        self.rail_mesh[1] = obj_visual(self, self.rails_dir + 'rail_turn_left.obj')
        self.rail_mesh[2] = obj_visual(self, self.rails_dir + 'rail_turn_right.obj')

        # grass meshes
        self.clutter_mesh = np.ones(3)
        self.clutter_mesh *= obj_visual(self, self.plants_dir + 'clutter_large.obj')


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

            # handle the rails
            spawn_id = self.rails[0].handle_rail_bounds(self.drone.state[-1][:2])
            spawn_id = -2 if spawn_id == -1 else spawn_id

            to_dereference = []
            for rail in self.rails[1:]:
                rail.handle_rail_bounds(self.drone.state[-1][:2], spawn_id)
                if not rail.exists:
                    to_dereference.append(rail)

            # handle the clutterring
            self.clutter.handle_rail_bounds(self.drone.state[-1][:2], spawn_id)

            # remove rails that are to be deleted
            self.rails = [rail for rail in self.rails if rail not in to_dereference]

            # render camera image
            self.drone.capture_image()
            return True


    def track_state(self) -> np.ndarray:
        # get the rail image of the main rail (rails[0])
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

