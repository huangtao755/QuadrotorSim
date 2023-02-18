
 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch as t
from torch.autograd import Variable

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

t.manual_seed(5)

sim_num = 20
x0 = np.array([2, -1])
epsilon = 1
Fre_V1_paras = 5


############################################################################################################
# Define Network
############################################################################################################

class Model(t.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.layer1 = np.zeros((32, input_dim))
        self.layer1[0:8, 0] = 1
        self.layer1[8:16, 1] = 1
        self.layer1[16:24, 2] = 1
        self.layer1[24:32, 3] = 1
        self.weight_matrix = np.random.random(size=(output_dim, 32))

    def forward(self, x):
        x = t.nn.functional.tanh(t.tensor(x))
        print(x.shape, '////////////////')
        x = t.tensor(self.layer1) @ x
        output = t.tensor(self.weight_matrix)@x
        return output

    def train(self, x, y):
        x = np.array([x])
        y = np.array([y])
        self.weight_matrix = np.linalg.pinv(x@x.T) @ x @ y.T
        self.weight_matrix = (x @ x.T)**(-1) @ x @ y.T
        return self.weight_matrix


############################################################################################################
# the policy based ADP with single net
############################################################################################################

class ADPSingleNet(object):
    def __init__(self, evn, replay_buffer,
                 learning_rate=0.01,
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
        self.gamma = 1

        'init critic net'
        self.critic_eval = Model(input_dim=self.state_dim, output_dim=1)
        self.critic_target = Model(input_dim=self.state_dim, output_dim=1)
        self.criterion = t.nn.MSELoss(reduction='mean')
        self.optimizerCritic = t.optim.Adam(self.critic_eval.parameters(),
                                            lr=learning_rate)
        self.batch = 32
        self.update_targetNet_freq = 20

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
        loss = []
        for train_index in range(learning_num):

            "Step one; get data"
            data = self.buffer.buffer_sample_batch(batch_size=self.batch)
            state = data[:, :self.state_dim]
            action = data[:, self.state_dim: self.state_dim + self.action_dim]
            reward = data[:, self.state_dim + self.action_dim]
            state_new = data[:, -self.state_dim:]

            state = t.tensor(state, dtype=t.float)
            action = t.tensor(action, dtype=t.float)
            reward = t.tensor(reward, dtype=t.float)
            state_new = t.tensor(state_new, dtype=t.float)
            "Step one; get data"
            for j in range(5):
                "Step two: calculate critic value"
                # critic_input = t.cat(state, action)
                critic_value = self.critic_eval(Variable(state))
                # print(critic_value, 'critic_eval')
                critic_value_next = self.critic_target(Variable(state_new))
                # print(critic_value_next, 'critic_target')
                "Step two: calculate critic value"

                "calculate the loss and update critic net"
                critic_loss = self.criterion(critic_value, Variable(reward * 0.01 + self.gamma * critic_value_next))
                self.optimizerCritic.zero_grad()
                critic_loss.backward()
                self.optimizerCritic.step()
                # print('the loss is %f' % critic_loss)
                # print('_______the Critic Net have updated for %d time_______' % train_index)
                "calculate the loss and update critic net"

                loss.append(float(critic_loss))

                if critic_loss < 0.005:
                    self.critic_target = self.critic_eval
                    print('update critic_target')
                    print('training finish')
                    return loss
            "update parameters of critic target net"
            if train_index % self.update_targetNet_freq == 0:
                self.critic_target = self.critic_eval
                print('update critic_target')
            "update parameters of critic target net"

        return loss

    def save_models(self):
        pass

    def load_models(self):
        pass

    def update_network_parameters(self):
        pass


if __name__ == "__main__":
    # fig = plt.figure(figsize=(10, 5))
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.title(" y = 10 * sin(x)")  # 标题
    # plt.xlabel("X轴")  # X轴标签
    # plt.ylabel("Y轴")  # Y轴标签
    # x, y = [], []
    # ani = FuncAnimation(fig, test,
    #                     frames=np.arange(0, 3, 10),
    #                     interval=5, blit=False, repeat=False)
    # plt.show()
    net = Model(4, 1)

    a = np.arange(0, 3, 0.1)
    b = a**2+np.sin(a)
    x1 = a
    x2 = a ** 0.2
    x3 = a ** 0.6
    x4 = a ** 2

    X =np.hstack((x1, x2, x3, x4))


    print(a)
    print(b)

    matrix = net.train(X, np.array(b))
    print(matrix)
    Y = net.forward(X)





