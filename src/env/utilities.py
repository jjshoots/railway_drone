import numpy as np
from pybullet_utils import bullet_client

def loadOBJ(p: bullet_client.BulletClient,
            fileName: str,
            meshScale=[1., 1., 1.],
            basePosition=[0., 0., 0.],
            baseOrientation=[0., 0., 0.]
            ):
    visualId = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=fileName,
        rgbaColor=[1, 1, 1, 1],
        specularColor=[0., 0., 0.],
        meshScale=meshScale
    )

    collisionId = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=fileName,
        meshScale=meshScale
    )

    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collisionId,
        baseVisualShapeIndex=visualId,
        basePosition=basePosition,
        baseOrientation=baseOrientation
    )
