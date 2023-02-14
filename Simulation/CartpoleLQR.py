from Evn.CartPole import CartPoleEnv
from Algorithm.LQR_control import *
import scipy.linalg as la
import numpy as np
import time


env = CartPoleEnv()
env.reset()
A, B = env.sys_matr()
print(A, B)

Q = np.array([[1000, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1000, 0],
     [0, 0, 0, 1]])
R = np.array([1])

k = lqr(A, B, Q, R, env.tau)
# k = np.array([-0.0000, -0.0100, -6.8420, -0.0000])
print(k)

for i in range(1000):
     action = float(-k@np.array(env.state).T)
     print(action)
     state, done = env.step(action)
     env.render()
     # print(action, type(action))
     time.sleep(0.01)
     if done:
        env.reset()
