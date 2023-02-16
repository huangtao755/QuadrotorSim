from Evn.CartPole import CartPoleEnv
from Algorithm.LQR_control import *
import scipy.linalg as la
from scipy.linalg import expm
from scipy.optimize import fsolve

import numpy as np
import time


class EventTrigger(object):
    def __init__(self, A, B, Q, R, K, P, xigma):
        self.A = A
        self.B = B
        self.Q = Q
        self.K = K
        self.P = P
        self.R = R

        self.xigma = xigma
        dim_b = int(self.K.shape[1])
        self.Phi1 = np.hstack(((self.xigma - 1)*self.Q, self.P@self.B@self.K))
        self.Phi2 = np.hstack((K.T@B.T@P, np.zeros((dim_b, dim_b))))
        self.Phi = np.vstack((self.Phi1, self.Phi2))

    def trigger_condition(self, xk, x):
        e = xk - x
        z = np.hstack((xk, e))
        condition = np.linalg.norm(z.T @ self.Phi @ z)
        print(condition)
        if condition >= 0:
            return True
        else:
            return False





