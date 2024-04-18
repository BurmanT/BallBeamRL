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
import math

class BallBeamBotPybullet:
    def __init__(self, client, gui, save_img, random_ball_location, setpoint, ball_threshold, delta, threshold):
        """
        Class to spawn in and control ball beam bot

        Parameters
        ----------
        client: id, (int)
        gui: render pybullet headless, (bool) 
        save_img: save frames while taking action, (bool)
        random_ball_location: location to drop ball on beam, (float)
        setpoint: location to balance ball, (float)
        ball_threshold: acceptable range for ball setpoint, (float)
        delta: angle to move beam at each action, (float)
        threshold: decimal place to check if two numbers are equal, (float)

        """
        self.client = client 
        self.gui = gui
        self.save_img = save_img
        self.setpoint = setpoint 
        self.ball_threshold = ball_threshold
        self.delta = delta
        self.threshold = threshold 
        # load ball and beam 
        beam_urdf_path = "ball_beam/resources/ballbeam/ballbeam.urdf"
        ball_urdf_path = "ball_beam/resources/ball/ball.urdf"
        self.bot = p.loadURDF(beam_urdf_path, [0, 0, 0], useFixedBase=True, physicsClientId=client)
        self.ball = p.loadURDF(ball_urdf_path, [random_ball_location, 0, 0.4], physicsClientId=client)
        
        self._hit_color = [1, 0, 0]
        self._miss_color = [0, 1, 0]
        self._ray_ids = None
    
    def get_ids(self):
        return self.bot, self.ball, self.client
    
    def within_setpoint(self):
        """
        Returns True if current ball position is within setpoint
        Returns False otherwise 

        """
        higher = self.setpoint + self.ball_threshold
        lower = self.setpoint - self.ball_threshold

        position = self.get_observation()[0]

        if(position >= lower and position <= higher):
            # print("PYBULLET WITHIN SETPOINT" + str(position))
            # print("Goal is " + str(self.setpoint)+ " LOWER IS "+ str(lower)+" HIGHER IS "+str(higher))
            return True
        
        return False


    def apply_action(self, action: int) -> None:
        """
        Performs action

        Parameters
        ---------- 
        action: value in [-1, 0, 1] (int)

        """
        # print("ACTION IN PYBULLET")
        # print(action)
    
        img_arr = []
        # keeps track of how many steps ball is within setpoint
        timesteps_setpoint = 0
        total_timesteps = 0
        current_angle = self.get_current_angle()
        #print("current agnle is ")
        #print(current_angle)
        #print("delta is " + str(self.delta))
        #print("action is ")
        #print(action)
        position = self.delta*action + current_angle
        #print("desired position is ")
        #print(position)
        # print("CURRENT ANGLE " + str(current_angle))
        # print("ANGLE TO TAKE " + str(position))
        # step through the simulation even if action is 0
        if(action == 0):
            for i in range(51):
                p.setJointMotorControl2(
                self.bot, 1, p.POSITION_CONTROL, targetPosition=position, force=100
                )
                p.stepSimulation()
                if((i % 20 == 0) and self.save_img):
                    px = self.camera()
                    img_arr.append(px)
                if(self.gui):
                    time.sleep(1./240)
                if(self.within_setpoint()):
                    timesteps_setpoint += 1
        else:
            for i in range(55):
                p.setJointMotorControl2(
                self.bot, 1, p.POSITION_CONTROL, targetPosition=position, force=100
                )
                p.stepSimulation()
                if((i % 20 == 0) and self.save_img):
                    px = self.camera()
                    img_arr.append(px)
                if(self.gui):
                    time.sleep(1./240)
                if(self.within_setpoint()):
                    timesteps_setpoint += 1
                if(math.isclose(position, self.get_current_angle(), abs_tol=self.threshold)):
                    #print("ANG EQUAL POS STOPPING AT i " + str(i))
                    total_timesteps = i
                    return img_arr, timesteps_setpoint, total_timesteps
            total_timesteps = 55
        # print("COMPLETED 100 TIMESTEPS")
        return img_arr, timesteps_setpoint, total_timesteps

    def get_current_angle(self) -> int:
        """
            Get current beam angle 
        """
        return p.getEulerFromQuaternion(p.getLinkState(self.bot, 2)[1])[1]


    def get_observation(self):
        """
            Uses lidar to get ball position 
            Returns: (ball position, beam angle)
        """

        sensor_range = 1

        ball_translation, _ = p.getBasePositionAndOrientation(
            self.ball
        )
        current_angle = self.get_current_angle()

        ray_angle = -1 * self.get_current_angle()
        ray_from = ball_translation
        ray_to = ball_translation + sensor_range * np.array([np.cos(ray_angle), 0, np.sin(ray_angle)])

        result = p.rayTest(ray_from, ray_to)
        position = np.array(result, dtype=object)[:, 2][0]

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
        # returns ball position, beam angle/ beam position
        return (position, current_angle)
    def camera(self):
        """
            Credit: Brennan 
            Produces top down camera image of environment
        """

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0],
            distance=70,
            yaw=45,
            pitch=-35,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=1, aspect=float(1920) / 1080, nearVal=0.1, farVal=100.0
        )
        (_, _, px, _, _) = p.getCameraImage(
            width=1920,
            height=1080,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        return px
    

if __name__=='__main__':
    bot = BallBeamBotPybullet(1, True)
    angle = p.addUserDebugParameter("Motor", -1, 1, 0)

    while(True):
        bot.apply_action(p.readUserDebugParameter(angle))
        #print(bot.get_observation())
        time.sleep(1. / 240)