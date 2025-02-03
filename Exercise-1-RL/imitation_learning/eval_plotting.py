# This file can be diregarded as it was not given, but I created it in order to print tensorboard results using matplotlib

import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator 

def smooth(scalars = [0], weight = 0.5): 
    end = scalars[0]  
    smoothed = []
    for point in scalars:
        smoothed_val = end * weight + (1 - weight) * point 
        smoothed.append(smoothed_val)                       
        end = smoothed_val                                 

    return smoothed

def plot_tensorboard_events(logdir="./tensorboard", tags=[], smoothing=0.8, xlabel="", ylabel="", title="", legend_title="", figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)

    for subdir in os.listdir(logdir):
        event_file = os.path.join(logdir, subdir, os.listdir(os.path.join(logdir, subdir))[0])

        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()

        for tag in tags:
            if tag in ea.Tags()['scalars']:
                steps = []
                values = []
                for scalar in ea.Scalars(tag):
                    steps.append(scalar.step)
                    values.append(scalar.value)
                smoothed_values = smooth(values, smoothing)
                ax.plot(steps, smoothed_values, label=os.path.basename(subdir) + "_" + tag)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title=legend_title)

    plt.show()

if __name__ == "__main__":

    plot_tensorboard_events(tags=["Accuracy"], xlabel="Epochs", ylabel="Accurcay", title="Accuracy %", smoothing=0.97)
    plot_tensorboard_events(tags=["Training_loss"], xlabel="Epochs", ylabel="Loss", title="Training Loss", smoothing=0.95)
