#!/usr/bin/env python3

"""
Script to move the ballbeam 
Based off of
    - https://github.com/erwincoumans/pybullet_robots/blob/master/turtlebot.py
    - https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-2-a1441b9a4d8e
Resources used for lidar: 
    - https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/batchRayCast.py
    - https://github.com/axelbr/racecar_gym/blob/master/racecar_gym/bullet/sensors.py
Resources used for camera:
    - https://www.programcreek.com/python/example/122153/pybullet.computeViewMatrixFromYawPitchRoll

"""

import pybullet as p
import numpy as np
import time
import random
import gym 
import math

import numpy as np
def create_uniform_grid(low, high, bins=(10, 10)):
    """Define a uniformly-spaced grid that can be used to discretize a space.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins along each corresponding dimension.

    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    # TODO: Implement this
    grids = []
    for l, h, n in zip(low, high, bins):
        grids.append(np.linspace(l, h, num=n, endpoint=False)[1:])

    return grids

def discretize(sample, grid):
    """Discretize a sample as per given grid.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.

    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    # TODO: Implement this
    digitized = []
    for s, g in zip(sample, grid):
        digitized.append(np.digitize(s, g))
    return digitized

class BallBeamBotPybullet:
    def __init__(self, gui, random_ball_pos):
        """class to spawn in and control bot
        """
        p.connect(p.GUI if gui else p.DIRECT)
        p.setGravity(0, 0, -9.8)
        #p.setRealTimeSimulation(1)
        p.setTimeStep(1. / 30)
        #p.setTimeStep(1/5)
        self.gui = gui
        
        #self.client = client 
        # beam_urdf_path = "/home/mulip/Ball-Beam-NEW/ball_beam/resources/ballbeam/ballbeam.urdf"
        # ball_urdf_path = "/home/mulip/Ball-Beam-NEW/ball_beam/resources/ball/ball.urdf"

        beam_urdf_path = "./ballbeam/ballbeam.urdf"
        ball_urdf_path = "./ball/ball.urdf"

        self.bot = p.loadURDF(beam_urdf_path, [0, 0, 0], useFixedBase=True)
        self.ball = p.loadURDF(ball_urdf_path, [random_ball_pos, 0, 0.4])
        # [0.25, 0, 0.4]
        # -0.4 to 0.25 to first val 

        #self.bot = p.loadURDF(beam_urdf_path, [0, 0, 0], useFixedBase=True, physicsClientId=client)
        #self.ball = p.loadURDF(ball_urdf_path, [0, 0, 0.6], physicsClientId=client)

        self._hit_color = [1, 0, 0]
        self._miss_color = [0, 1, 0]
        self._ray_ids = None
    
    def get_ids(self):
        return self.bot, self.ball, self.client


    def apply_action(self, action: int) -> None:
        """
        Performs action

        :param action: -1, 0, 1
        """
        print("Chosen action is ")
        print(action)
        # current_angle = self.get_current_angle()
        # print("Current angle is ")
        # print(current_angle)
        # # displacement for angle 
        # delta = 0.01
        
        if(action != 0):
            # position = delta*action + current_angle 
            # print("Position before rounding ")
            # print(position)
            # position = round(position, 4)
            for i in range(150):
                # print(i)
                # print("Position to take is ")
                # print(position)
                p.setJointMotorControl2(
                self.bot, 1, p.POSITION_CONTROL, targetPosition=action, force=100
                )
                p.stepSimulation()
                time.sleep(1. / 240)
                ang = self.get_current_angle()
                #print("Observation angle before round " + str(ang))
                if(math.isclose(action, ang, abs_tol = 1e-04)):
                    print("ANG AND POS EQUAL at iteration " + str(i))
                    return
            print("COMPLETED 150 timesteps")
                
        # print("Action is ")
        # print(action)
        # p.setJointMotorControl2(
        # self.bot, 1, p.POSITION_CONTROL, targetPosition=action, force=1000
        # )
        # print("current angle is ")
        # print(self.get_current_angle())
        
        

    def get_current_angle(self) -> int:
        """
        """
        return round(p.getEulerFromQuaternion(p.getLinkState(self.bot, 2)[1])[1],4)


    def get_observation(self) -> float:
        """simulate lidar measurement
        """

        sensor_range = 1

        ball_translation, _ = p.getBasePositionAndOrientation(
            self.ball
        )
        current_angle = self.get_current_angle()
        ray_angle = -1 * current_angle

        #print(current_angle)
        #print(ray_angle)
        ray_from = ball_translation
        ray_to = ball_translation + sensor_range * np.array([np.cos(ray_angle), 0, np.sin(ray_angle)])

        result = p.rayTest(ray_from, ray_to)
        # print(result)
        # print(np.array(result, dtype=object)[:, 2])
        position = np.array(result, dtype=object)[:, 2]
        position = round(position[0], 4)
        # print(position)
        # print("CURRENT ANGLE " + str(ray_angle))
        # key=input("STOP")

        hitObjectUid = result[0][0]

        if self.gui:
            if self._ray_ids is None:
                self._ray_ids = p.addUserDebugLine(ray_from, ray_to, self._miss_color)

            if (hitObjectUid < 0):
                p.addUserDebugLine(
                    ray_from,
                    ray_to,
                    self._miss_color,
                    replaceItemUniqueId=self._ray_ids
                )
            else:
                hit_location = result[0][3]
                p.addUserDebugLine(
                    ray_from,
                    hit_location,
                    self._hit_color,
                    replaceItemUniqueId=self._ray_ids
                )

        # returns ball position, velocity, beam angle
        return position, current_angle
        #return np.array(result, dtype=object)[:, 2], current_angle, dict()
        #return np.array([result, ray_angle])

low = [0, -0.2]
high = [0.75, 0.2]
grid = create_uniform_grid(low, high)  # [test]

if __name__=='__main__':
    num = -0.4
    bot = BallBeamBotPybullet(True, num)
    angle = p.addUserDebugParameter("Motor", -0.2, 0.2, 0)
    
    while(True):
        read = p.readUserDebugParameter(angle)
        bot.apply_action(read)
        p.stepSimulation()
        pos, ang = bot.get_observation()
        obs = [pos, ang]
        obs = np.array(obs)
        print("OBSERVATION IS POSITION, CURRENT ANGLE: ")
        print(obs)
        print("DISCRETIZED OBSERVATION IS POSITION, CURRENT ANGLE: ")
        print(discretize(obs, grid))
        time.sleep(1. / 240)