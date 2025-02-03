from __future__ import print_function

import sys
sys.path.append("../") 

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

from utils import *
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space 
    #    using action_to_id() from utils.py.

    img_shape = np.shape(X_train[0])

    X_train = np.array([rgb2gray(x) for x in X_train])
    X_valid = np.array([rgb2gray(x) for x in X_valid])
    X_train = X_train[:, np.newaxis]
    X_valid = X_valid[:, np.newaxis]
    y_train = np.array([action_to_id(y) for y in y_train])
    y_valid = np.array([action_to_id(y) for y in y_valid])

    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).

    if history_length > 0:
        dummy_images = np.zeros((history_length, 1, img_shape[0], img_shape[1]))
        X_train = np.concatenate((dummy_images, X_train), axis=0)
        X_valid = np.concatenate((dummy_images, X_valid), axis=0)

        X_train = [X_train[i:i+history_length+1] for i in range(len(X_train)-history_length)]
        X_train = np.array(X_train).reshape(-1, history_length+1, img_shape[0], img_shape[1])

        X_valid = [X_valid[i:i+history_length+1] for i in range(len(X_valid)-history_length)]
        X_valid = np.array(X_valid).reshape(-1, history_length+1, img_shape[0], img_shape[1])
    else:
        X_train = np.array(X_train)
        X_valid = np.array(X_valid)

    y_train = np.array(y_train)
    y_valid = np.array(y_valid)

    return X_train, y_train, X_valid, y_valid

def train_model(X_train, y_train, X_valid, y_valid, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")


    # TODO: specify your agent with the neural network in agents/bc_agent.py 
    # agent = BCAgent(...)
    agent = BCAgent(learning_rate=lr)
    tensorboard_eval = Evaluation(store_dir=tensorboard_dir, name="training", stats=["Training_loss","Accuracy"])

    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    # 
    # training loop
    # for i in range(n_minibatches):
    #     ...
    #     for i % 10 == 0:
    #         # compute training/ validation accuracy and write it to tensorboard
    #         tensorboard_eval.write_episode_data(...)
    
    running_loss = 0.0
    validation_accuracy = 0.0

    for epochs in range(n_minibatches):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            loss = agent.update(X_batch=X_batch,y_batch=y_batch)
            running_loss += loss

            if i % (1000) == 0:    
                print(f'epochs:{epochs}, i:{i}, loss = {loss.item():.4f}')
                tensorboard_eval.write_episode_data(episode=epochs, eval_dict={'Training_loss': running_loss/10})
                running_loss = 0.0
        for i in range(0, len(X_valid), batch_size):
            X_valid_batch = X_valid[i:i+batch_size]
            y_valid_batch = y_valid[i:i+batch_size]
            val_output = agent.predict(X_valid_batch)
            for i in range(len(val_output)):
                action_id = val_output[i]
                action_id = action_id.argmax().item()
                if action_id == y_valid_batch[i]:
                    validation_accuracy += 1
        validation_accuracy = (validation_accuracy/len(X_valid)) * 100
        print(f'Validation accuracy = {validation_accuracy:.4f}')
        tensorboard_eval.write_episode_data(episode=epochs, eval_dict={'Accuracy': validation_accuracy})

    # tensorboard --logdir=tensorboard

    # TODO: save your agent 
    model_dir = agent.save(os.path.join(model_dir, "agent_history3_cnn_ndata.pt"))
    print("Model saved in file: %s" % model_dir)

if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=2)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=130, batch_size=100, lr=1e-4)

