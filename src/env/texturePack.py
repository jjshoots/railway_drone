import os
import glob
import random

from pybullet_utils import bullet_client

class TexturePack():
    def __init__(self, p: bullet_client.BulletClient, path):
        self.p = p
        self.texture_paths = glob.glob(os.path.join(path, '**', '*.jpg'), recursive=True)

    def get_random_texture(self) -> int:
        texture_path = self.texture_paths[random.randint(0, len(self.texture_paths) - 1)]
        return self.p.loadTexture(texture_path)

