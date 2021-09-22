import cv2
import math
import time
import numpy as np

import matplotlib.pyplot as plt

from environment import *
from texturePack import *

def main():
    # initialize the environment
    env = Environment(render=True)
    tex = TexturePack('dtd')
    env.drone.set_mode(4)

    """ SIMULATE """
    # simulate
    for i in range (100000):

        if i < 500:
            env.step()
            pass
        elif i > 500 and i < 2000:
            env.step()
            env.drone.setpoint = np.array([0, 0, 0, 2])
        else:
            if env.step():
                env.drone.setpoint = env.get_flight_target()

        # every 1000 time steps (~ 4 seconds), change the texture of the floor
        if i % 1000 == 2:
            env.changeVisualShape(env.planeId, -1, textureUniqueId=tex.get_random_texture())

            tex_id = tex.get_random_texture()
            env.change_rail_texture(tex_id)

    """ DISCONNECT """
    # disconnect
    env.disconnect()

if __name__ == "__main__":
    main()
