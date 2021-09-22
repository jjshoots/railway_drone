import numpy as np

from PID import *

from pybullet_utils import bullet_client

class RailObject():
    def __init__(
            self, p: bullet_client.BulletClient,
            start_pos: np.ndarray,
            start_orn: np.ndarray,
            obj_file: str,
            parent=None
            ):

        self.p = p

        ROT = 0.105

        if 'straight' in obj_file:
            self.start_pos = start_pos
            self.start_orn = start_orn
            self.end_pos = self.start_pos + np.array([20.24*np.sin(-start_orn[-1]), 20.24*np.cos(start_orn[-1]), 0])
            self.end_orn = self.start_orn + np.array([0, 0, 0])
            self.base_pos = self.start_pos + np.array([10.12*np.sin(-start_orn[-1]), 10.12*np.cos(start_orn[-1]), 0])
            self.base_orn = self.start_orn
        elif 'left' in obj_file:
            rotation = ROT + start_orn[-1]
            self.start_pos = start_pos
            self.start_orn = start_orn
            self.end_pos = self.start_pos + np.array([20.24*np.sin(-rotation), 20.24*np.cos(rotation), 0])
            self.end_orn = self.start_orn + np.array([0, 0, ROT*3])
            self.base_pos = self.start_pos + np.array([10.12*np.sin(-start_orn[-1]), 10.12*np.cos(start_orn[-1]), 0])
            self.base_orn = self.start_orn
        elif 'right' in obj_file:
            rotation = -ROT + start_orn[-1]
            self.start_pos = start_pos
            self.start_orn = start_orn
            self.end_pos = self.start_pos + np.array([20.24*np.sin(-rotation), 20.24*np.cos(rotation), 0])
            self.end_orn = self.start_orn + np.array([0, 0, -ROT*3])
            self.base_pos = self.start_pos + np.array([10.12*np.sin(-start_orn[-1]), 10.12*np.cos(start_orn[-1]), 0])
            self.base_orn = self.start_orn

        # a straight rail has length 20.24
        self.Id = self.loadOBJ(
            obj_file,
            basePosition=self.base_pos,
            baseOrientation=self.p.getQuaternionFromEuler(self.base_orn)
        )

        # linked = [parent, child]
        self.linked = [None, None]
        self.linked[0] = parent


    def add_child(self, obj_file):
        """ adds a single child to the end of the rail """
        self.linked[1] = RailObject(self.p, self.end_pos, self.end_orn, obj_file, self)


    def delete(self, dir: int):
        """ deletes self and all connected links in dir, 0 for parent, 1 for child """
        deleted = np.array([])
        node = self
        # traverse to the end of the chain
        while node.linked[dir] is not None:
            node = node.linked[dir]

        while True:
            # delete the node model
            self.p.removeBody(node.Id)
            deleted = np.append(deleted, node.Id)

            # move up the chain by a step and dereference
            if node.linked[1-dir] is not None:
                node = node.linked[1-dir]
                node.linked[dir] = None

                if node.Id == self.linked[1-dir].Id:
                    break
            else:
                node = None     # this is very bad
                break

        return deleted, node


    def get_end(self, dir: int):
        """ gets the end of the track, depending on dir, 0 for parent, 1 for child """
        node = self
        while node.linked[dir] is not None:
            node = node.linked[dir]

        return node


    def loadOBJ(self, fileName, meshScale=[1., 1., 1.], basePosition=[0., 0., 0.], baseOrientation=[0., 0., 0.]):
        visualId = self.p.createVisualShape(
            shapeType=self.p.GEOM_MESH,
            fileName=fileName,
            rgbaColor=[1, 1, 1, 1],
            specularColor=[0., 0., 0.],
            meshScale=meshScale
        )

        collisionId = self.p.createCollisionShape(
            shapeType=self.p.GEOM_MESH,
            fileName=fileName,
            meshScale=meshScale
        )

        return self.p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collisionId,
            baseVisualShapeIndex=visualId,
            basePosition=basePosition,
            baseOrientation=baseOrientation
        )