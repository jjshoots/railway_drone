import math
import numpy as np

from pybullet_utils import bullet_client
from env.utilities import *

class RailObject():
    def __init__(self, p:bullet_client.BulletClient,
                 start_pos: np.ndarray,
                 start_orn: np.ndarray,
                 rails_dir: str
                 ):

        self.p = p
        self.rails_dir = rails_dir

        start_pos = np.array([0, 0, 0])
        start_orn = np.array([0.5*math.pi, 0, 0])
        rail = RailSingle(p, start_pos, start_orn, self.rails_dir+'rail_straight.obj')

        self.Ids = np.array([rail.Id])
        self.head = rail.get_end(0)
        self.tail = rail.get_end(1)
        self.tex_id = None


    def handle_rail_bounds(self, drone_xy):
        dis2head = np.sum((self.head.base_pos[:2] - drone_xy) ** 2) ** 0.5
        dis2tail = np.sum((self.tail.base_pos[:2] - drone_xy) ** 2) ** 0.5

        # delete the head if it's too far and get the new one
        if dis2head > 20:
            deleted, self.head = self.head.delete(0)
            self.Ids = [id for id in self.Ids if id not in deleted]

        # create new tail if it's too near
        if dis2tail < 40:
            rand_idx = np.random.randint(0, 3)
            obj_file = self.rails_dir + 'rail_straight.obj'
            # rand_idx = 1
            if rand_idx == 1:
                obj_file = self.rails_dir + 'rail_turn_left.obj'
            if rand_idx == 2:
                obj_file = self.rails_dir + 'rail_turn_right.obj'
            self.tail.add_child(obj_file)
            self.tail = self.tail.get_end(1)
            self.Ids = np.append(self.Ids, self.tail.Id)
            if self.tex_id is not None:
                self.p.changeVisualShape(self.tail.Id, -1, textureUniqueId=self.tex_id)


    def change_rail_texture(self, tex_id):
        self.tex_id = tex_id
        for id in self.Ids:
            self.p.changeVisualShape(id, -1, textureUniqueId=self.tex_id)



class RailSingle():
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
        else:
            print('IMPORTING UNKNOWN RAIL OBJECT')
            self.start_pos = start_pos
            self.start_orn = start_orn
            self.end_pos = start_pos
            self.end_orn = start_orn
            self.base_pos = start_pos
            self.base_orn = start_orn

        # a straight rail has length 20.24
        self.Id = loadOBJ(
            self.p,
            obj_file,
            basePosition=self.base_pos,
            baseOrientation=self.p.getQuaternionFromEuler(self.base_orn)
        )

        # linked = [parent, child]
        self.linked = [None, None]
        self.linked[0] = parent


    def add_child(self, obj_file):
        """ adds a single child to the end of the rail """
        self.linked[1] = RailSingle(self.p, self.end_pos, self.end_orn, obj_file, self)


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

