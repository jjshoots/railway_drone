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

from dronectrl import *

class Environment(bullet_client.BulletClient):
    def __init__(self, render=True, camera_Hz=24, frame_size=(128, 128)):
        super().__init__(p.GUI if render else p.DIRECT)

        # default physics looprate is 240 Hz
        self.period = 1. / 240.
        self.now = time.time()

        self.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.render = render
        self.reset()

        """ CAMERA """
        self.proj_mat = self.computeProjectionMatrixFOV(fov=50.0, aspect=1.0, nearVal=0.1, farVal=255.)
        self.rgbImg = self.depthImg = self.segImg = None
        self.rel_cam_Hz = int(240 / camera_Hz)
        self.frame_size = frame_size

        """ CONSTRUCT THE WORLD LAST """
        # the world
        self.planeId = self.loadURDF(
            "samurai.urdf",
            useFixedBase=True
        )

        # spawn rail
        start_pos = [0, 0, 0]
        start_orn = self.getQuaternionFromEuler([.5 * math.pi, 0, 0])
        self.railId = self.loadOBJ(
            "tracks_obj/rail_straight.obj",
            basePosition=start_pos,
            baseOrientation=start_orn
        )

        # spawn drone
        self.drone = DroneCtrl(self)


    def reset(self):
        self.resetSimulation()
        self.setGravity(0, 0, -9.81)
        self.step_count = 0


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

        self.drone.rpm2forces(np.array([1., 1., 1., 1.]))

        self.stepSimulation()
        self.step_count += 1

        if self.step_count % self.rel_cam_Hz == 0:
            self.capture_image()
            return True


    @property
    def view_mat(self):
        # get the state of the camera on the robot
        camera_state = self.getLinkState(self.drone.Id, 4)

        # pose and rot
        position = camera_state[0]
        rotation = np.array(self.getMatrixFromQuaternion(camera_state[1])).reshape(3, 3)

        # compute camera up vector using rot_mat
        up_vector = np.matmul(rotation, np.array([0., 0., 1.]))

        # target position is 1000 units ahead of camera relative to the current camera pos
        target = np.dot(rotation, np.array([1000, 0, 0])) + np.array(position)

        return self.computeViewMatrix(
            cameraEyePosition=position,
            cameraTargetPosition=target,
            cameraUpVector=up_vector
        )


    def capture_image(self):
        _, _, self.rgbImg, self.depthImg, self.segImg = self.getCameraImage(
            width=self.frame_size[1],
            height=self.frame_size[0],
            viewMatrix=self.view_mat,
            projectionMatrix=self.proj_mat
        )


    def loadOBJ(self, fileName, meshScale=[1., 1., 1.], basePosition=[0., 0., 0.], baseOrientation=[0., 0., 0.]):
        visualId = self.createVisualShape(
            shapeType=self.GEOM_MESH,
            fileName=fileName,
            rgbaColor=[1, 1, 1, 1],
            specularColor=[0., 0., 0.],
            meshScale=meshScale
        )

        collisionId = self.createCollisionShape(
            shapeType=self.GEOM_MESH,
            fileName=fileName,
            meshScale=meshScale
        )

        return self.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collisionId,
            baseVisualShapeIndex=visualId,
            basePosition=basePosition,
            baseOrientation=baseOrientation
        )
