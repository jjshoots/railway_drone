import os
import glob
import random

import pybullet as p

class TexturePack():
    def __init__(self, path):
        self.texture_paths = glob.glob(os.path.join(path, '**', '*.jpg'), recursive=True)

    def get_random_texture(self) -> int:
        texture_path = self.texture_paths[random.randint(0, len(self.texture_paths) - 1)]
        return p.loadTexture(texture_path)

