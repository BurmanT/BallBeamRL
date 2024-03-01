import gym
import ball_beam
from PG import SimplePG
import matplotlib.pyplot as plt
import pickle 
import numpy as np
import time
from PIL import Image
import pandas as pd
import seaborn as sns
import os 
import random

"""
Directory saved scheme: 
	./DIRNAME/run_x/model.pkl

"""
def save_fig(dirname, run_num, final, random_seed, agent, total_sum_reward_tracker, total_timestep_setpoint_tracker, total_timestep_tracker):
	"""
	Saves the model and its figures 
    
    Parameters
    ----------
	dirname: Name of directory to save model and figures in (str)
	run_num: run number, (int)
	final: if its the sum of all runs, (bool)
    agent: PG model 
	total_sum_reward_tracker: list of rewards over the episodes 
	total_timestep_setpoint_tracker: list of timesteps at each episode 
	total_timestep_tracker: total timesteps at each episode 
        
    """
	if(final):
		run_name = "./" + dirname + "/final"
	else: 
		run_name = "./" + dirname + "/run" + str(run_num) + "_" + str(random_seed)
	
	if not os.path.exists(dirname):
   		os.makedirs(dirname)
	
	if not os.path.exists(run_name):
   		os.makedirs(run_name)

	# save the model if its not the final 		   
	if not final:
		modelname = run_name + "/model.pkl"
		# save the model
		pickle.dump(agent, open(modelname,'wb'))

	episodes = len(total_sum_reward_tracker)
	# get percent of timesteps in setpoint
	perc = []
	
	for i in range(len(total_timestep_setpoint_tracker)):
		if(total_timestep_tracker[i] == 0):
			perc.append(0)
		else:
			perc.append((total_timestep_setpoint_tracker[i] / total_timestep_tracker[i]) * 100)
	# print("PERC IS")
	# print(perc)
	iterations = range(0, episodes, 1)
	df = pd.DataFrame(list(zip(iterations, total_sum_reward_tracker,perc)), columns=['Iterations','Rewards', 'Setpoint'])
	df['rewards_rolling_avg'] = df.Rewards.rolling(100).mean()
	df['setpoint_rolling_avg'] = df.Setpoint.rolling(100).mean()

	plt.figure()
	sns.lineplot(x='Iterations',y='Rewards', data=df, label="Rewards History")
	sns.lineplot(x='Iterations',y='rewards_rolling_avg', data=df, label="Rewards Rolling Avg")
	plt.ylabel('Average Return'), plt.xlabel('Episode')
	save_name = run_name +'/training_rewards.png'
	rewards_name = run_name + '/training_rewards_run' + str(run_num) + '.pkl'
	timesteps_name = run_name + '/setpoint_run' + str(run_num) + '.pkl'
	plt.savefig(save_name)
	pickle.dump(total_sum_reward_tracker, open(rewards_name,'wb'))
	pickle.dump(perc, open(timesteps_name,'wb'))
	
	plt.figure()
	sns.lineplot(x='Iterations',y='Setpoint', data=df, label="Timesteps Near Setpoint")
	sns.lineplot(x='Iterations',y='setpoint_rolling_avg', data=df, label="Timesteps Near Setpoint Rolling Avg")
	plt.ylabel('Average Timesteps'), plt.xlabel('Episode')
	save_name = run_name +'/setpoint.png'
	plt.savefig(save_name)

def model_checkpoint(dirname, run, random_seed, agent, env, timesteps, ep):
	"""
		Save the model with random_seed, current e value
		Parameters
		----------
		dirname: Name of directory to save model
		run: run number, (int)
		random_seed: random seed, (int)
		agent: model
		env: env
		timesteps: num timesteps in an episode, (int)
		ep: episode of model checkpoint, (int)
	"""
	run_name = "./" + dirname + "/run" + str(run) + "_" + str(random_seed)
	modelname = run_name + "/model_checkpoint_" + str(ep) + ".pkl"
	# save the model
	pickle.dump(agent, open(modelname,'wb'))
	env.eval = True 
	rewards = []
	action_space = [-1, 0, 1]
	for e in range(20):
		obs = env.reset()
		reward_sum = 0
		for i in range(timesteps):
			action = agent.process_step(obs,False)
			# obs is [position, velocity, angle]
			obs, reward, done, info = env.step(action_space[action])
			reward_sum += reward

			if(done):
				rewards.append(reward_sum)
	setpoint, total = env.get_eval()
	arr = np.array(rewards)
	return np.average(arr)

def main_dqn(dirname, run, random_seed, gui, episodes, timesteps, debug):
	"""
    Run one epoch 
    
    Parameters
    ----------
	dirname: Name of directory to save model and figures in (str)
	run: run number, (int)
	random_seed: random seed, (int)
    gui: Turn gui on or off (bool)
    episodes: Num episodes to run (int)
    timesteps: Num timesteps in one episode (int)
	debug: prints obs, action and reward, (bool)
        
    """
	env = gym.make('BallBeam-v0', gui=gui, timesteps=timesteps)
	actionCnt = env.action_space.n
	action_space = [-1, 0, 1]
	D = env.observation_space.shape[0] # how many input neurons
	NUM_HIDDEN = 10
	GAMMA = 0.95
	LEARNING_RATE = 1e-3
	DECAY_RATE = 0.99
	MAX_EPSILON = 0.1

	agent = SimplePG(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
	agent.set_explore_epsilon(MAX_EPSILON)
	reward_sum = 0
	running_reward = None
	rewards_history = []
	# keeps track of reward during evaluation
	total_eval_rewards = []
	for e in range(episodes):
		obs = env.reset()
		# print("RESETTING OBS IS [pos, vel, ray_angle]")
		# print(obs)
		#key=input("stop here")
		reward_sum = 0
		for t in range(timesteps):
			action = agent.process_step(obs,True)
			#print("ACTION CHOSEN IS "+ str(action))

			# obs is [position, velocity, angle]
			obs, reward, done, info = env.step(action_space[action])

			if(debug):
				print("TIMESTEP "+str(t)+" ACTION "+str(action))
				print("OBS: [position, velocity, angle]")
				print(obs)
				print("REWARD = "+str(reward) + " DONE = " + str(done))
			
			agent.give_reward(reward)
			reward_sum += reward
			if done is True:
				# print(t)
				# key=input("stop here")
				running_reward = reward_sum if running_reward is None else running_reward * 0.95 + reward_sum * 0.05
				rewards_history.append(running_reward)
				
				print('ep %f: resetting env. episode reward total was %f. episode time steps was %f. running mean: %f' % (e, reward_sum, t, running_reward))
				
				agent.finish_episode()
				# update after every k episodes
				if e % 5 == 0:
					agent.update_parameters()
				break
		# checkpoint: evaluate the model 
		# if(e % 250 == 0):
		# 	eval_reward = model_checkpoint(dirname, run, random_seed, agent, env, timesteps, e)
		# 	total_eval_rewards.append(eval_reward)
	
	total_sum_reward_tracker, total_timestep_setpoint_tracker, total_timestep_tracker = env.close()
	save_fig(dirname, run, False, random_seed, agent, rewards_history, total_timestep_setpoint_tracker, total_timestep_tracker)
	return total_sum_reward_tracker, total_timestep_setpoint_tracker, total_timestep_tracker
	#return rewards_history, total_timestep_setpoint_tracker, total_timestep_tracker

def get_model(dirname, run, random_seed):
	objects = []
	dir = "./" + dirname + "/" + "run" + str(run) + "_" + str(random_seed) + "/model.pkl"
	with (open(dir, "rb")) as openfile:
		while True:
			try:
				objects.append(pickle.load(openfile))
			except EOFError:
				break
	return objects[0]

def test(dirname, run, gui, episodes, timesteps, random_seed):
	"""
	Parameters
    ----------
	dirname: directory with all results, (str)
	run: run number, (int)
	gui:
	episodes: 
	"""
	env = gym.make('BallBeam-v0', gui=gui, timesteps=timesteps, save_img=True)
	obs = env.reset()
	# key=input("stop here")
	action_space = [-1, 0, 1]
	agent = get_model(dirname, run, random_seed)
	agent.reset()
	agent._exploration = False
	img_array = []
	print("TESTING MODEL... ")
	for i in range(episodes):
		for j in range(timesteps):
			action = agent.process_step(obs, False)
			obs, reward, done, info = env.step(action_space[action])
			frame = info["camera"]
			img_array.extend(frame)
			# frame = env.camera()
			# img_array.append(frame)
			if done is True:
				# Create the GIF
				gif_name = "./" + dirname + "/" + "run" + str(run) + "_" + str(random_seed) + "/gif" + str(i) + ".gif"
				imgs = [Image.fromarray(img) for img in img_array]
				# duration is the number of milliseconds between frames; this is 40 frames per second
				imgs[0].save(gif_name, save_all=True, append_images=imgs[1:], duration=50, loop=0)
				print("SAVED GIF NUM " + str(i))
				obs = env.reset()
				img_array = []
	env.close()

def run_all(dirname, num_runs, gui, episodes, timesteps, debug, test_episodes):
	"""
	dirname: name of directory, (str)
	num_runs: total number of runs, (int)
	gui: boolean 
	episodes: episodes in each epoch, (int)
	timesteps: time steps in each episode, (int)
	debug: debug mode, (bool) 
	test_episodes: number of gifs to produce from trained model, (int)
	"""
	rewards = []
	time_steps_setpoint = []
	time_steps_total = []

	for i in range(num_runs):
		random_seed = random.randint(1, 20)
		r, setpoint_t, total_t  = main_dqn(dirname, i, random_seed, gui, episodes, timesteps, debug)
		rewards.append(r)
		time_steps_setpoint.append(setpoint_t)
		time_steps_total.append(total_t)
		#test(dirname, i, gui, test_episodes, timesteps, random_seed)
	
	df_r = pd.DataFrame(np.array(rewards))
	df_t = pd.DataFrame(np.array(time_steps_setpoint))
	df_tot = pd.DataFrame(np.array(time_steps_total))

	avg_r = df_r.mean(axis=0)
	avg_t = df_t.mean(axis=0)
	avg_tot = df_tot.mean(axis=0)

	avg_r_list = avg_r.values.tolist()
	avg_t_list = avg_t.values.tolist()
	avg_tot_list = avg_tot.values.tolist()

	save_fig(dirname,None, True, None, None, avg_r_list, avg_t_list, avg_tot_list)
	print("COMPLETED ALL RUNS ")

if __name__ == "__main__":
	dirname = "RESULTS_DIR"
	num_runs = 10
	gui = False
	episodes = 10000
	timesteps = 100
	debug = False
	test_episodes = 3
	run_all(dirname, num_runs, gui, episodes, timesteps, debug, test_episodes)
	#test(dirname, 4, gui, test_episodes, timesteps, randomseed)
             
