import math
import time
import numpy as np
import torch.nn as nn

import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client
from environment import *

def main():
    # initialize the environment
    env1 = Environment()
    texture_paths = glob.glob(os.path.join('dtd', '**', '*.jpg'), recursive=True)

    """ SIMULATE """
    # simulate
    for i in range (100000):
        env1.step()

        # every 1000 time steps (~ 4 seconds), change the texture of the floor
        if i % 1000 == 999:
            # get a random texture
            random_texture_path = texture_paths[random.randint(0, len(texture_paths) - 1)]
            # load a texture into pb
            textureId = env1.loadTexture(random_texture_path)
            # change the texture of the object
            env1.changeVisualShape(env1.planeId, -1, textureUniqueId=textureId)

    """ DISCONNECT """
    # disconnect
    env1.disconnect()

if __name__ == "__main__":
    main()
