#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch as t
from torch.autograd import Variable


t.manual_seed(5)

sim_num = 20
x0 = np.array([2, -1])
epsilon = 0.0001
Fre_V1_paras = 5


############################################################################################################
# Define Network
############################################################################################################

class Model(t.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.lay1 = t.nn.Linear(input_dim, 10, bias=False)
        self.lay1.weight.data.normal_(0, 0.5)
        self.lay2 = t.nn.Linear(10, output_dim, bias=False)
        self.lay2.weight.data.normal_(0, 0.5)

    def forward(self, x):
        layer1 = self.lay1(x)
        layer1 = t.nn.functional.relu(layer1)
        output = self.lay2(layer1)
        return output


############################################################################################################
# the policy based ADP with single net
############################################################################################################

class ADPSingleNet(object):
    def __init__(self, evn, replay_buffer,
                 learning_rate=0.005,
                 state_dim=12,
                 action_dim=4):
        """

        :param evn:
        :param replay_buffer:
        :param learning_rate:
        """
        'init evn'
        self.evn = evn
        self.buffer = replay_buffer

        self.state_dim = state_dim
        self.state = np.zeros(state_dim)
        self.action_dim = action_dim
        self.action = np.zeros(action_dim)
        self.gx = None

        'reward parameter'
        # self.Q = t.tensor(evn.Q)
        # self.R = t.tensor(evn.R)
        # self.gamma = 1

        'init critic net'
        self.critic_eval = Model(input_dim=self.state_dim, output_dim=1)
        self.critic_target = Model(input_dim=self.state_dim, output_dim=1)
        self.criterion = t.nn.MSELoss(reduction='mean')
        self.optimizerCritic = t.optim.SGD(self.critic_eval.parameters(), lr=learning_rate)
        self.batch = 32
        self.update_targetNet_freq = 10

    def choose_action(self, state):
        """
        :brief:
        :param state:  the current state
        :return:       the optimal action
        """
        "Step one: calculate the gradient of the critic net"
        state = t.tensor(state, dtype=t.float, requires_grad=True)
        critic_value = self.critic_eval(state)
        critic_value.backward()
        critic_grad = state.grad
        "Step one: calculate the gradient of the critic net"

        "Step two: calculate the action according to the HJB function and system dynamic function"
        value = t.mm(t.pinverse(self.R), self.gx)
        action = - t.mm(value, critic_grad.T) / 2
        "Step two: calculate the action according to the HJB function and system dynamic function"
        return action

    def learn(self, learning_num):
        """

        :param learning_num:
        :return:
        """
        for train_index in range(learning_num):

            "Step one; get data"
            data = self.buffer.buffer_sample_batch(batch_size=self.batch)
            state, action, state_new, reward = data[0], data[1], data[2], data[3]
            print(state[0], action[0], state_new[0], reward[0], 'state')
            state = t.tensor(state, dtype=t.float)
            # action = t.tensor(action, dtype=t.float)
            reward = t.tensor(reward, dtype=t.float)
            state_new = t.tensor(state_new, dtype=t.float)
            "Step one; get data"

            "Step two: calculate critic value"
            critic_value = self.critic_eval(Variable(state))
            critic_value_next = self.critic_target(Variable(state_new))
            "Step two: calculate critic value"

            "calculate the loss and update critic net"
            critic_loss = self.criterion(critic_value, Variable(reward + self.gamma * critic_value_next))
            self.optimizerCritic.zero_grad()
            critic_loss.backward()
            self.optimizerCritic.step()
            print('_______the Critic Net have updated for %d time_______' % train_index)
            print('the loss is %f' % critic_loss)
            "calculate the loss and update critic net"

            "update parameters of critic target net"
            if train_index % self.update_targetNet_freq == 0:
                pass
            "update parameters of critic target net"

    def save_models(self):
        pass

    def load_models(self):
        pass

    def update_network_parameters(self):
        pass
