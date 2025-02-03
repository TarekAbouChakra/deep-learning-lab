import numpy as np
import torch
import torch.optim as optim
from agent.replay_buffer import ReplayBuffer

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, gamma=0.95, batch_size=64, epsilon=0.95, tau=0.01, lr=1e-4,
                 history_length=0, capacity = 1e6, epsilon_min = 0.01, epsilon_decay = 0.99):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            gamma: discount factor of future rewards.
            batch_size: Number of samples per batch.
            tau: indicates the speed of adjustment of the slowly updated target network.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
            lr: learning rate of the optimizer
        """
        # setup networks
        self.Q = Q.cuda()
        self.Q_target = Q_target.cuda()
        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer(history=history_length, capacity=capacity)

        # parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)

        self.num_actions = num_actions

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """

        # TODO:
        # 1. add current transition to replay buffer
        # 2. sample next batch and perform batch update: 
        #       2.1 compute td targets and loss 
        #              td_target =  reward + discount * max_a Q_target(next_state_batch, a)
        #       2.2 update the Q network
        #       2.3 call soft update for target network
        #           soft_update(self.Q_target, self.Q, self.tau)


        # if (len(self.replay_buffer._data.states) < self.batch_size):
        #     return

        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)

        state_batch, action_batch, next_state_batch, reward_batch, done_batch = self.replay_buffer.next_batch(self.batch_size)

        state_batch = torch.tensor(state_batch, dtype=torch.float).cuda()
        action_batch = torch.tensor(action_batch, dtype=torch.long).cuda()
        reward_batch = torch.tensor(reward).cuda()
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float).cuda()
        done_batch = torch.tensor(done_batch, dtype=torch.bool).cuda()

        batch_indices = np.arange(self.batch_size, dtype=np.int64)
        Q_pred = self.Q.forward(state_batch)[batch_indices, action_batch]

        Q_next = torch.max(self.Q_target.forward(next_state_batch), dim=1)[0]

        td_return = reward_batch + self.gamma * Q_next * done_batch

        loss = self.loss_function(Q_pred, td_return)

        # update Q network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        soft_update(self.Q_target, self.Q, self.tau)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            # TODO: take greedy action (argmax)
            # action_id = ...
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float).cuda()
                q_values = self.Q(state_tensor)
            action_id = torch.argmax(q_values).item()
        else:
            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work. 
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            # action_id = ...

            # Action ID for cartpole
            # Please uncomment this part and comment the one having probabilities and action_id below for cartpole
            # action_id = np.random.choice(self.num_actions)

            # Action ID for carracing
            # Please uncomment this part and comment the one action_id above for carracing
            probabilities = [0.075, 0.15, 0.15, 0.5, 0.125]
            action_id = np.random.choice(a = self.num_actions, p=probabilities)

        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name)

    def load(self, file_name):
        self.Q.load_state_dict(torch.load(file_name))
        self.Q_target.load_state_dict(torch.load(file_name))
