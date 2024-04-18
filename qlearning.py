import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ball_beam
import os 
import pickle
from PIL import Image
import sys
import seaborn as sns

def create_uniform_grid(low, high, bins=(10, 10, 10)):
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

class QLearningAgent:
    """Q-Learning agent that can act on a continuous state space by discretizing it."""

    def __init__(self, env, state_grid, alpha=0.02, gamma=0.99,
                 epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=505):
        """Initialize variables, create grid for discretization."""
        # Environment info
        self.env = env
        self.state_grid = state_grid
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)  # n-dimensional state space
        #print(self.state_size)
        self.action_size = self.env.action_space.n  # 1-dimensional discrete action space
        self.seed = np.random.seed(seed)
        #print("Environment:", self.env)
        #print("State space size:", self.state_size)
        #print("Action space size:", self.action_size)
        
        # Learning parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = self.initial_epsilon = epsilon  # initial exploration rate
        self.epsilon_decay_rate = epsilon_decay_rate # how quickly should we decrease epsilon
        self.min_epsilon = min_epsilon
        
        # Create Q-table
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        #print(self.q_table)
        #print("Q table size:", self.q_table.shape)
        #print("action size w comma ")
        #print(self.state_size)
        #print((self.action_size,))
        #print((self.state_size + (self.action_size,)))

    def preprocess_state(self, state):
        """Map a continuous state to its discretized representation."""
        
        return tuple(discretize(state, self.state_grid))

    def reset_episode(self):
        """Reset variables for a new episode."""
        # Gradually decrease exploration rate
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)

        # Decide initial action
        # self.last_state = self.preprocess_state(state)
        # self.last_action = np.argmax(self.q_table[self.last_state])
        # return self.last_action
    
    def reset_exploration(self, epsilon=None):
        """Reset exploration rate used when training."""
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

    def choose_action(self, state, mode='train'):
        """Pick next action based on mode """
        #print("IN CHOOSE ACTION state is ")
        #print(state)
        state = self.preprocess_state(state)
        if mode == 'test':
            # Test mode: Simply produce an action
            #action = np.argmax(self.q_table[state])
            action = np.random.choice(np.flatnonzero(self.q_table[state] == self.q_table[state].max()))
        else:
            # Exploration vs. exploitation
            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                #print("Random action chosen")
                # Pick a random action
                action = np.random.randint(0, self.action_size)
                #print("action is " + str(action))
            else:
                # Pick the best action from Q table
                # break ties between same max value 
                action = np.random.choice(np.flatnonzero(self.q_table[state] == self.q_table[state].max()))
                #print("Action chosen is " + str(action))
                #print("q table and q table of specific state is ")
                #print(self.q_table)
                #print(self.q_table[state])
                #print("done with action function")

            # Train mode (default): Update Q table, pick next action
            # Note: We update the Q table entry for the *last* (state, action) pair with current state, reward
            #self.q_table[self.last_state + (self.last_action,)] += self.alpha * \
            #    (reward + self.gamma * max(self.q_table[state]) - self.q_table[self.last_state + (self.last_action,)])

        # Roll over current state, action for next step
        self.last_state = state
        self.last_action = action
        return action
    
    def update(self, new_state, reward):
        """ update internal Q table (when mode != 'test'). """
        #print("in update function. q table is ")
        #print(self.q_table)
        #print("last state is ")
        #print(self.last_state)
        #print("last action is ")
        #print(self.last_action)
        new_state = self.preprocess_state(new_state)
        #print("new state is ")
        #print(new_state)
        #print("q table at new state is ")
        #print(self.q_table[new_state])
        #print("max val of q table in new state is ")
        #print(max(self.q_table[new_state]))
        self.q_table[self.last_state + (self.last_action,)] += self.alpha * (reward + self.gamma * max(self.q_table[new_state]) - self.q_table[self.last_state + (self.last_action,)])
         

def run(agent, env, bin, num_episodes=20000, mode='train'):
    """Run agent in given reinforcement learning environment and return scores."""
    scores = []
    max_avg_score = -np.inf
    action_space = [-1, 0, 1]
    state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=bin)
    #print(state_grid)
    for i_episode in range(1, num_episodes+1):
        # Initialize episode
        state = env.reset()
        agent.reset_episode()
        #print("state and discretized state is ")
        #print(state)
        #print(tuple(discretize(state, state_grid)))
        # reset episode function separate from choose action
        
        total_reward = 0
        done = False
        i = 0
        # Roll out steps until done
        while not done:
            i += 1
            #print("TIME STEP "+ str(i))
            action = agent.choose_action(state)
            # override action
            #action = input("Enter Action: ")
            #action = int(action)
            state, reward, done, info = env.step(action_space[action])
            agent.update(state, reward)
            #print(reward)
            #print("state and discretized state is ")
            #print(state)
            #print(tuple(discretize(state, state_grid)))
            total_reward += reward
            #action = agent.act(state, reward, done, mode)
            # override acrion

        # Save final score
        scores.append(total_reward)
        
        # Print episode stats
        if mode == 'train':
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score

            if i_episode % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
                #sys.stdout.flush()
    #print(agent.q_table)
    return scores

def test(agent, timesteps, dirname, reward_type, num_episodes=3, mode='test'):
    """Run agent in given reinforcement learning environment and return scores."""
    env = gym.make('BallBeam-v0', gui=False, timesteps=timesteps, save_img=True, reward_type=reward_type)
    #obs = env.reset()
	# key=input("stop here")
    action_space = [-1, 0, 1]
    img_arr = []
    print("NOW SAVING GIF")
    for i_episode in range(num_episodes):
        print("EPISODE " + str(i_episode))
        # Initialize episode
        state = env.reset()
        #agent.reset_episode()
        total_reward = 0
        done = False

        # Roll out steps until done
        for j in range(timesteps):
            action = agent.choose_action(state)
            state, reward, done, info = env.step(action_space[action])
            total_reward += reward
            
            #print(frame)
            frame = info["camera"]
            img_arr.extend(frame)

            if done is True:
                #print(img_arr)
                # Create the GIF
                gif_name = dirname + "/gif_" + str(i_episode) + ".gif"
                print(gif_name)
                imgs = [Image.fromarray(img) for img in img_arr]
                # duration is the number of milliseconds between frames; this is 40 frames per second
                imgs[0].save(gif_name, save_all=True, append_images=imgs[1:], duration=50, loop=0)
                print("SAVED GIF NUM " + str(i_episode))
                img_arr = []
    env.close()

def plot_scores(scores, dirname, rolling_window=100, mode='train'):
    """Plot scores and optional rolling mean using specified window."""
    plt.figure()
    plt.plot(scores)
    title = mode + '_reward'
    plt.title(title)
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean)
    save_name = dirname + '/' + mode + '_rewards.png'
    plt.savefig(save_name)
    #return rolling_mean

def plot_setpoint(dirname, setpoint_t, total_t):
    episodes = len(total_t)
	# get percent of timesteps in setpoint
    perc = []
    for i in range(len(total_t)):
        if(total_t[i] == 0):
            perc.append(0)
        else:
            perc.append((setpoint_t[i] / total_t[i]) * 100)
    iterations = range(0, episodes, 1)
    df = pd.DataFrame(list(zip(iterations,perc)), columns=['Iterations','Setpoint'])
    df['setpoint_rolling_avg'] = df.Setpoint.rolling(100).mean()
    plt.figure()
    sns.lineplot(x='Iterations',y='Setpoint', data=df, label="Timesteps Near Setpoint")
    sns.lineplot(x='Iterations',y='setpoint_rolling_avg', data=df, label="Timesteps Near Setpoint Rolling Avg")
    plt.ylabel('Average Timesteps'), plt.xlabel('Episode')
    save_name = dirname +'/setpoint.png'
    plt.savefig(save_name)

def get_model(dirname):
	objects = []
	dir = "./" + dirname + "/model.pkl"
	with (open(dir, "rb")) as openfile:
		while True:
			try:
				objects.append(pickle.load(openfile))
			except EOFError:
				break
	return objects[0]

def main(num_runs, episodes, rew, bin_ind):
    dirname = "QLearningResults"
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    if(bin_ind == '0'):
        bin = (5, 5, 5)
        run_name = "./" + dirname + "/bin_" + bin_ind 
    elif(bin_ind == '1'):
        bin = (20, 20, 20)
        run_name = "./" + dirname + "/bin_" + bin_ind 
    else:
        bin = (10, 10, 10)
        run_name = "./" + dirname + "/bin_2"

    if(rew == '0'):
        reward_type = [10, 0, 0]
        run_name += "_rew_" + rew
    elif(rew == '1'):
        reward_type = [10, 0, -1]
        run_name += "_rew_" + rew
    else:
        reward_type=[10, 5, -1]
        run_name += "_rew_2"
    
    if not os.path.exists(run_name):
        os.makedirs(run_name)

    all_reward = []
    time_steps_setpoint = []
    time_steps_total = []
    # Create a grid to discretize the state space
    random_seeds = [10, 20, 30, 40, 50]
    for i in range(num_runs):
        gui = False
        timesteps = 10
        env = gym.make('BallBeam-v0', gui=gui, timesteps=timesteps, reward_type=reward_type)
        state_grid = create_uniform_grid(env.observation_space.low, env.observation_space.high, bins=bin)
        q_agent = QLearningAgent(env, state_grid, seed=random_seeds[i])
        scores = run(q_agent, env, bin, num_episodes=episodes)
        all_reward.append(scores)
        modelname = run_name + "/model_" + str(i) +".pkl"
        # save the model
        pickle.dump(q_agent, open(modelname,'wb'))
        print("MODEL SAVED")
        r, setpoint_t, total_t = env.close()
        time_steps_setpoint.append(setpoint_t)
        time_steps_total.append(total_t)
    # plot rewards 
    all_reward = np.array(all_reward)
    all_reward = np.mean(all_reward, axis=0)
    plot_scores(all_reward, run_name)
    plot_setpoint(run_name, setpoint_t, total_t)
    
    test(q_agent, timesteps, run_name, reward_type)

if __name__ == "__main__":
    """
    python qlearning.py [rew_type] [bin_type]

    where rew_type is '0', '1' or anything else:
    '0': [10, 0, 0]
    '1': [10, 0, -1]
    default: [10, 5, 1]

    where bin_type is '0', '1' or anything else:
    '0': [5, 5, 5]
    '1': [20, 20, 20]
    default: [10, 10, 10]
    """
    if(len(sys.argv) > 2):
        rew = sys.argv[1]
        bins = sys.argv[2]
    num_runs = 5
    episodes = 5
    main(num_runs, episodes, rew, bins)
    