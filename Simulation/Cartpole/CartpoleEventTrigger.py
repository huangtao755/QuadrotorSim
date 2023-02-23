from Evn.CartPole import CartPoleEnv
from Algorithm.LQR_control import *
from Algorithm.Event_trigger import *
import scipy.linalg as la
from scipy.linalg import expm
from scipy.optimize import fsolve

import numpy as np
import time


env = CartPoleEnv('env')
env.reset()

envk = CartPoleEnv('envk')
envk.reset(init_state=env.state)

A, B = env.sys_matr()

Q = np.array([[100, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 100, 0],
     [0, 0, 0, 1]])
R = np.array([1])

K, P = lqr(A, B, Q, R, env.tau)

F = np.vstack((np.hstack((A + B@K, B@K)), np.hstack((-A - B@K, -B@K))))
C = np.hstack((np.eye(4), np.zeros((4, 4))))

F_e = expm(F)
FT_e = expm(F.T)

print('A----------------:\n', A, '\nB----------------:\n', B)
print('K----------------:\n', K, '\nP----------------:\n', P)
print('F----------------:\n', F, '\nC----------------:\n', C)
print('Q----------------:\n', Q, '\nR----------------:\n', R)
print(expm(F.T))


print(A.shape, 'A 的维度')
print(B.shape, 'B 的维度')
print(C.shape, 'C 的维度')
print(P.shape, 'P 的维度')
print(F.shape, 'F 的维度')

Trigger = EventTrigger(A=A, B=B, Q=Q, R=R, K=K, P=P, xigma=0)

action = float(-K@np.array(env.state).T)
actionk = copy(action)
xk = envk.state
for i in range(2000):
     action = float(-K@np.array(env.state).T)
     state, done = env.step(action)
     # env.render()

     if Trigger.trigger_condition(np.array(xk), np.array(envk.state)):
          print('ok')
          xk = envk.state
          actionk = action
     statek, donek = envk.step(actionk)
     envk.render()

     time.sleep(0.01)
     if done:
          env.reset()
     if donek:
          envk.reset()
env.fig_show(i=1)
envk.fig_show(i=2)
plt.show()