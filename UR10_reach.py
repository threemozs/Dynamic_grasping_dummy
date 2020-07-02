"""
UR10 robot reaches 5 randomly places targets (defined by the variable LOOPS).
This script contains examples of:
    - Linear (IK) paths.
    - Scene manipulation (creating an object and moving it).
"""

from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.ur10 import UR10
from pyrep.objects.shape import Shape  # modify scene
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError
import numpy as np
import math
import time

LOOPS = 30
SCENE_FILE = join(dirname(abspath(__file__)), 'UR10_reach_003.ttt')
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)  # lunch the ttt file
pr.start()
agent = UR10()  # take a look --

# We could have made this target in the UR10_reach.ttt scene, but lets create one dynamically

# position_min, position_max = [0.8, -0.2, 1.0], [1.5, 0.7, 1.0]

starting_joint_positions = [-1.2234,-0.3290,1.6263,1.1871,1.9181,1.5707]       # the robot joints parameters, what are these params??--
print('ee_pos_init:', agent.get_tip().get_position())


for i in range(LOOPS):
    target = Shape.create(type=PrimitiveShape.CUBOID,
                          size=[0.05, 0.05, 0.4],
                          color=[1.0, 0.1, 0.1],
                          static=False,
                          position=[1.0, 0.2, 0.95],
                          # position=[1.0, 0.2, 1.002],
                          respondable=True)

    goal_ = Shape.create(type=PrimitiveShape.SPHERE,  # the cuboid
                         # size=[0.05, 0.05, 0.6],  # increased the height by 0.2
                         size=[0.02, 0.02, 0.02],
                         mass=0.0,
                         smooth=False,
                         color=[1.0, 0.1, 0.1],
                         static=True, respondable=True)
    goal_.set_position(np.array([1.0, 0.7, 1.5]))

    # start_ = Shape.create(type=PrimitiveShape.SPHERE,  # the cuboid
    #                      # size=[0.05, 0.05, 0.6],  # increased the height by 0.2
    #                      size=[0.02, 0.02, 0.02],
    #                      mass=0.0,
    #                      smooth=False,
    #                      color=[1.0, 0.1, 0.1],
    #                      static=True, respondable=True)
    # start_.set_position(np.array([1.0, 0.2, 1.0]))


    # Reset the arm at the start of each 'episode'
    agent.set_joint_positions(starting_joint_positions)

    print('target pos;', target.get_position())

    ee_init_pos = np.array([1, 0.2, 0.95])
    # Get a path to the target (rotate so z points down)
    try:
        path = agent.get_path(
            position=ee_init_pos, euler=[-2.18, 3.1415, 0])  # generate path given position and euler angles. NOTE: sometime the end-eff knock over the obj, why?
        # path = agent.get_path(
        #     position=pos, euler=[3.1415, 3.1415,
        #                          0])  # generate path given position and euler angles. NOTE: sometime the end-eff knock over the obj, why?
    except ConfigurationPathError as e:
        print('Could not find path')   # print error
        continue

    # Step the simulation and advance the agent along the path
    done = False
    while not done:
        done = path.step()  # how does step works?
        pr.step()

        agent_joint_positions = agent.get_joint_target_positions()
        print('agent_joint_positions:', agent_joint_positions)

    print('ee_pos:', agent.get_tip().get_position())

    time.sleep(3)


    print('Reached target %d!' % i)
    print('agent orientation:', agent.get_orientation())
    ee = agent.get_tip()
    ee_ori = ee.get_orientation()
    print('ee_ori:', ee_ori)
    target.remove()



pr.stop()  # Stop the simulation
pr.shutdown()  # Close the application