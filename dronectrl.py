import os
import glob
import random
import math
import time
import numpy as np
import torch.nn as nn

import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

class DroneCtrl():
    def __init__(self, p: bullet_client.BulletClient):
        # default physics looprate is 240 Hz
        self.p = p
        self.period = 1. / 240.

        # spawn robocar
        start_pos = [1, 0, 1]
        start_orn = self.p.getQuaternionFromEuler([0, 0, 0])
        self.Id = self.p.loadURDF(
            "models/primitive_car/car.urdf",
            basePosition=start_pos,
            baseOrientation=start_orn,
            useFixedBase=False
        )

        for i in range(self.p.getNumJoints(self.Id)):
            print(self.p.getJointInfo(self.Id, i))

        """ DRONE STUFF """
        # the joint IDs corresponding to motorID 1234
        # as detailed in Quadrotor X in PX4
        self.motor_map = np.array([0, 3, 2, 1])
        self.thr_coeff = np.array([[0., 0., 0.3]])
        self.tor_dir = np.array([[1.], [1.], [-1.], [-1.]])
        self.tor_coeff = np.array([[0., 0., 1.]])


    def rpm2forces(self, rpm):
        rpm = np.expand_dims(rpm, axis=1)
        thrust = rpm * self.thr_coeff
        torque = rpm * self.tor_coeff * self.tor_dir

        for idx, thr, tor in zip(self.motor_map, thrust, torque):
            self.p.applyExternalForce(self.Id, idx, thr, [0., 0., 0.], self.p.LINK_FRAME)
            self.p.applyExternalTorque(self.Id, idx, tor, self.p.LINK_FRAME)
