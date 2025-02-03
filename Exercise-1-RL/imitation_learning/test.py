from __future__ import print_function

import sys
sys.path.append("../") 

from datetime import datetime
import numpy as np
import gym
import os
import json

from agent.bc_agent import BCAgent
from utils import *

# Please make sure history length is the same as that of the model

# Note: running this now will use CNN history 2

def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    history_length = 2
    dummy_history = np.zeros((1, np.shape(state)[0], np.shape(state)[1]))
    grayState_history = []
    
    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events() 

    while True:
        
        # TODO: preprocess the state in the same way than in your preprocessing in train_agent.py
        #    state = ...

        grayState_save = np.array(rgb2gray(state)[np.newaxis, :, :])
        grayState = np.array(rgb2gray(state)[np.newaxis, :, :])
        
        if history_length > 0:
            if step == 0:
                for i in range(history_length):
                    grayState = np.concatenate((dummy_history, grayState), axis=0)
            elif history_length > step:
                for i in range(len(grayState_history)):
                    grayState = np.concatenate((grayState_history[i], grayState),axis=0)
                for i in range(history_length - step):
                    grayState = np.concatenate((dummy_history, grayState), axis=0)
            else:
                for i in range(len(grayState_history)):
                    grayState = np.concatenate((grayState_history[i], grayState), axis=0)

        if len(grayState_history) == history_length:
            grayState_history.pop()
        grayState_history.insert(0, grayState_save)

        # TODO: get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration
        # a = ...
        grayState = grayState[np.newaxis, :, :, :]
        action_id = agent.predict(grayState)
        action_id = action_id.argmax().item()
        action = id_to_action(action_id, max_speed=1)

        next_state, r, done, info = env.step(action)   
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test

    # TODO: load agent
    # agent = BCAgent(...)
    # agent.load("models/bc_agent.pt")

    agent = BCAgent()
    agent.load("models/agent_history2_cnn_ndata.pt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
