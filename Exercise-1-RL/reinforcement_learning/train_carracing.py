# export DISPLAY=:0 

import sys
sys.path.append("../") 

from agent.networks_car import *

import numpy as np
import gym
from tensorboard_evaluation import *
from utils import EpisodeStats, rgb2gray
from utils import *
from agent.dqn_agent import DQNAgent


def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, rendering=False, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events() 

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(history_length + 1, 96, 96)
    
    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)
        action_id = agent.act(state=state, deterministic=deterministic)
        action = id_to_action(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal: 
                 break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(history_length + 1, 96, 96)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if terminal or (step * (skip_frames + 1)) > max_timesteps : 
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, history_length=0, model_dir="./models_carracing", tensorboard_dir="./tensorboard"):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train_carracing"), name="training_race" , stats=["episode_reward", "straight", "left", "right", "accel", "brake", "avg_reward"])
    
    skip_frames = 2
    max_timesteps = 150
    rendering =  False

    for i in range(num_episodes):
        print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)

        stats = run_episode(env, agent, rendering=rendering, deterministic=False, do_training=True, max_timesteps=max_timesteps, 
        skip_frames=skip_frames, history_length=history)

        tensorboard.write_episode_data(episode= i, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                      "straight" : stats.get_action_usage(STRAIGHT),
                                                      "left" : stats.get_action_usage(LEFT),
                                                      "right" : stats.get_action_usage(RIGHT),
                                                      "accel" : stats.get_action_usage(ACCELERATE),
                                                      "brake" : stats.get_action_usage(BRAKE)
                                                      })

        print("episode_reward: ", stats.episode_reward)

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to 
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        # if i % eval_cycle == 0:
        #    for j in range(num_eval_episodes):
        #       ...

        if (i+1) % eval_cycle == 0:
            avg_reward = 0
            for j in range(num_eval_episodes):
                eval_stats = run_episode(env, agent, rendering=rendering, deterministic=True, do_training=False, 
                max_timesteps=max_timesteps, history_length=history)
                avg_reward += eval_stats.episode_reward
            avg_reward /= num_eval_episodes
            print("The average reward is: ",avg_reward)
            tensorboard.write_episode_data(episode=i, eval_dict={"avg_reward": avg_reward})
            
        if (i+1) % 150 == 0:
            if max_timesteps != 1000:
                max_timesteps +=25
            elif max_timesteps >= 1000:
                max_timesteps /= 2
        
        if max_timesteps >= 400:
            rendering =  True

        # store model.
        if i % eval_cycle == 0 or (i >= num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent.ckpt")) 

    # tensorboard.close_session()

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

if __name__ == "__main__":

    num_eval_episodes = 5
    eval_cycle = 20

    env = gym.make('CarRacing-v0').unwrapped

    # TODO: Define Q network, target network and DQN agent
    # ...

    action_space = 5
    history = 3

    Q_network = CNN(action_dim=action_space, history_length=history)
    target_net = CNN(action_dim=action_space, history_length=history)
    agent = DQNAgent(Q=Q_network, Q_target=target_net, num_actions=action_space, history_length=history, capacity=25000,
     epsilon=0.99, epsilon_decay=0.999, epsilon_min=0.1)
    train_online(env, agent, history_length=history, num_episodes=10000, model_dir="./models_carracing", tensorboard_dir="./tensorboard")

