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

A = np.array([[0.00000000e+00, 1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
             [0.00000000e+00, -1.69230769e-02, -1.80923077e+00, 0.00000000e+00],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
             [0.00000000e+00, 3.07692308e-02, 2.11076923e+01, 0.00000000e+00]])

B = np.array([[0.],
              [1.69230769],
              [0.],
              [-3.07692308]])

k, p = lqr(A, B, Q, R, env.tau)
print(k, p)
# k = np.array([-28.44008447,-21.4366359, -84.77012238, -18.65928996])
F = np.array([[A + B@k, B@k],
              [-A - B@k, -B@k]])

# for i in range(1000):
#      action = float(-k@np.array(env.state).T)
#      print(action)
#      state, done = env.step(action)
#      env.render()
#      # print(action, type(action))
#      time.sleep(0.01)
#      if done:
#         env.reset()
