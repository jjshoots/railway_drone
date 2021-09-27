import time
import numpy as np
import pybullet as p
import pybullet_data

def loadOBJ(fileName, meshScale=[1., 1., 1.], basePosition=[0., 0., 0.], baseOrientation=[0., 0., 0.]):
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

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0, 0, -9.81)
p.step_count = 0

""" CONSTRUCT THE WORLD """
p.planeId = p.loadURDF(
    "plane.urdf",
    useFixedBase=True
)

# spawn drone
loadOBJ('../models/vehicles/hector.obj')

while True:
    time.sleep(1/240.)
    p.stepSimulation()
