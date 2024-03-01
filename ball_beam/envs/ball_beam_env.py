import gym
import pybullet as p
import numpy as np
import math
import matplotlib.pyplot as plt
import random

from ball_beam.resources.ballbeam_pybullet import BallBeamBotPybullet

import time
class BallBeamEnv(gym.Env):
    '''
    Gym environment for BallBeamBot
    '''

    metadata = {"render.modes": ["human"]}

    def __init__(self, gui=False, timesteps=100, setpoint=0.4, random_ball_pos=False, ball_drop=0.1, reward_type=[10, 5, -1], save_img=False, ball_threshold=0.15, delta=0.02, threshold=1e-04, gravity=-1.0):
        '''
        Setup Gym environment, start pybullet and call reset

        Parameters
        ----------
        gui: determines whether pybullet is run headlessly, (bool)
        timesteps: timesteps per episode, (int)
        setpoint: where ball should balance on beam, (float)
        random_ball_pos: if ball should be dropped in random position every episode (bool)
        ball_drop: where ball should be dropped on beam if random_ball_pos is False (float)
        reward_type: list of reward to provide agent, ex:[10, 5, -1] 
                    10 if agent is within setpoint
                    5 if agent is going correct direction
                    -1 else
        save_img: save frames while taking action, (bool)
        ball_threshold: acceptable range for ball setpoint, (float)
        delta: angle to move beam at each action, (float)
        threshold: decimal place to check if two numbers are equal, (float)
        gravity: gravity in pybullet (float)
        '''
        # action space is -1, 0, 1
        self.action_space = gym.spaces.Discrete(3, start=-1)
        # position of ball, change in position, angle of beam 
        self.observation_space = gym.spaces.box.Box(
            low=np.array([0, -1, -0.2], dtype=np.float32),
            high=np.array([1, 1, 0.2], dtype=np.float32),
        )
        # used to set random position of ball
        self.np_random, _ = gym.utils.seeding.np_random()
        self.random_ball_pos = random_ball_pos 
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        self.gui = gui
        self.steps_per_episode = timesteps
        # Reduce length of episodes for RL algorithms 
        p.setTimeStep(1/30, self.client)
        self.save_img = save_img
        self.ball_threshold = ball_threshold
        self.delta = delta
        self.threshold = threshold 
        self.ball_drop = ball_drop
        self.reward_type = reward_type
        self.goal = setpoint
        self.gravity = gravity
        # ball location at each step
        self.prev_dist_to_goal = None
        self.current_position = None
        self.velocity = None

        self.bot = None

        self.done = False
        # reward received in one episode 
        self.episode_reward_tracker = []
        # reward received over all episodes 
        self.total_sum_reward_tracker = []
        # total num timesteps ball is in setpoint over all episodes 
        self.total_timestep_setpoint_tracker = []
        # total num of internal time steps over all episodes 
        self.total_timestep_tracker = []
        
        # keep track of num time steps 
        self.ts = 0
        # keeps track of internal timesteps ball is in setpoint in one episode
        self.setpoint_timesteps = 0
        # keeps track of total internal time steps in one episode 
        self.total_internal_timesteps = 0
        # checks if you are evaluating 
        self.eval = False 
        self.eval_setpoint_ts = []
        self.eval_total_internal_timesteps = []
        self.reset()

    def step(self, action):
        '''
        Take action and return observation

        Parameters
        ---------- 
        action: action to take, (int)
        '''
        img_arr, tsteps_setpoint, tsteps_total = self.bot.apply_action(action)
        #print("In STEP ACTION setpoint tsteps is " + str(tsteps_setpoint) + " " + str(tsteps_total))
        obs = self.bot.get_observation()
        new_position = obs[0]
        ray_angle = obs[1]
        # print("OBS of BALL " + str(new_position))
        # print("OBS of BEAM " + str(ray_angle))

        # velocity = new_position - old_position
        self.velocity = new_position - self.current_position
        # set current_position to be new position
        self.current_position = new_position
    
        dist_to_goal = self.current_position - self.goal
        higher = self.goal + self.ball_threshold
        lower = self.goal - self.ball_threshold
        # print("POSITION OF BALL IS " + str(new_position))
        # print("Goal is " + str(self.goal)+ " LOWER IS "+ str(lower)+" HIGHER IS "+str(higher))
        # print("Previous Distant to goal is " + str(self.prev_dist_to_goal))
        # print("Distance to goal is " + str(dist_to_goal))
        if(new_position >= lower and new_position <= higher):
            # print("WITHIN SETPOINT " + str(new_position))
            # print("Goal is " + str(self.goal)+ " LOWER IS "+ str(lower)+" HIGHER IS "+str(higher))
            reward = self.reward_type[0]
        elif(abs(self.prev_dist_to_goal) > abs(dist_to_goal)):
            # print("Previous Distant to goal is " + str(self.prev_dist_to_goal))
            # print("Distance to goal is " + str(dist_to_goal))
            # print("GETTING CLOSER")
            reward = self.reward_type[1]
        else:
            reward = self.reward_type[2]
        #print("REWARD IS " + str(reward))
        
        self.prev_dist_to_goal = dist_to_goal
        self.ts += 1
        # if currently not evaluating the model then continue to update rewards 
        self.setpoint_timesteps += tsteps_setpoint
        self.total_internal_timesteps += tsteps_total
        if not self.eval:
            self.episode_reward_tracker.append(reward)
        # Done if passed timestep limit
        if (self.ts >= self.steps_per_episode):
            # print("Timesteps completed ")
            # print("TS IS "+ str(self.ts))
            # print("STEPS PER EPISDOE IS "+ str(self.steps_per_episode))
            self.done = True

        if self.done: 
            self.collect_statistics()
            self.ts = 0
            self.setpoint_timesteps = 0
            self.total_internal_timesteps = 0
            
        obs = np.array([self.current_position, self.velocity, ray_angle])
        info = {"camera": img_arr}

        return (obs, reward, self.done, info)
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        '''
        Reset robot
        '''
        p.resetSimulation(self.client)
        p.setGravity(0, 0, self.gravity)
        # if spawn ball in random location on beam 
        if(self.random_ball_pos):
            # -0.4 and 0.25 are limits for where the ball can be
            random_ball_position = self.np_random.uniform(-0.4, 0.25)
            self.bot = BallBeamBotPybullet(self.client, self.gui, self.save_img, random_ball_position, self.goal, self.ball_threshold, self.delta, self.threshold)
        else:
            # Reload the beam   
            self.bot = BallBeamBotPybullet(self.client, self.gui, self.save_img, self.ball_drop, self.goal, self.ball_threshold, self.delta, self.threshold)
        
        for i in range (10):
            obs = self.bot.get_observation()
            position = obs[0]
            # set current_position to be new position
            self.current_position = position
            p.stepSimulation()
            if(self.gui):
                time.sleep(1./240.)
        obs = self.bot.get_observation()
        position = obs[0]
        ray_angle = obs[1]
        
        self.velocity = float(-1)
        # set current_position to be new position
        self.current_position = position
        self.prev_dist_to_goal = self.current_position - self.goal
        
        self.done = False
        self.ts = 0
        self.setpoint_timesteps = 0
        self.total_internal_timesteps = 0
        # print("IN RESET FUNCTION, current position, velocity and ray angle")
        # print(self.current_position)
        # print(self.velocity)
        # print(ray_angle)
        obs = np.array([self.current_position, self.velocity, ray_angle])
        return obs

    def close(self):
        '''
        Close 
        '''
        p.disconnect(self.client)
        return self.total_sum_reward_tracker, self.total_timestep_setpoint_tracker, self.total_timestep_tracker

    def collect_statistics(self) -> None:
        '''
        collect statistics function is used to record total sum and total timesteps per episode
        '''
        if not self.eval:
            self.total_sum_reward_tracker.append(sum(self.episode_reward_tracker))
            self.episode_reward_tracker = []
            self.total_timestep_setpoint_tracker.append(self.setpoint_timesteps)
            self.total_timestep_tracker.append(self.total_internal_timesteps)
        else:
            self.eval_setpoint_ts.append(self.setpoint_timesteps)
            self.eval_total_internal_timesteps.append(self.total_internal_timesteps)
    
    def get_eval(self):
        prev_eval_setpoint = self.eval_setpoint_ts
        prev_eval_tot = self.eval_total_internal_timesteps
        self.eval_setpoint_ts = []
        self.eval_total_internal_timesteps = []
        self.eval = False
        return prev_eval_setpoint, prev_eval_tot

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
    