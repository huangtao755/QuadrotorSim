#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import *
from math import *
import matplotlib.pyplot as plt
import scipy.linalg as la
import time

def solve_DARE(A, B, Q, R):
    """
    :brief:     solve a discrete time_Algebraic Riccati equation (DARE)
    :param A:
    :param B:
    :param Q:
    :param R:
    :return:
    """
    """
    
    """
    P = Q
    mapiter = 500
    eps = 0.000001
    print(A.T @ P @ A)
    print(A.T @ P @ B)
    print(R + B.T @ P @ B)
    for i in range(mapiter):
        Pn = A.T @ P @ A - (A.T @ P @ B) @ linalg.pinv(R + B.T @ P @ B) @ (B.T @ P @ A) + Q
        if (abs(Pn - P)).max() < eps:
            P = Pn
            break
        P = Pn
    return Pn

def dlqr(A, B, Q, R):
    """
    :brief:         Solve the discrete time lqr controller.
                    P[k+1] = A P[k] + B u[k]
                    cost = sum P[k].T*Q*P[k] + u[k].T*R*u[k]
    :param self:
    :param A:
    :param B:
    :param Q:
    :param R:
    :return:
    """

    # first, try to solve the ricatti equation
    P = solve_DARE(A, B, Q, R)
    print(P, 'P')
    # compute the LQR gain
    K = linalg.pinv(B.T @ P @ B + R) @ B.T @ P @ A
    return K


def lqr(A, B, Q, R, ts):
    """
    :brief:         Solve the discrete time lqr controller.
                    P[k+1] = A P[k] + B u[k]
                    cost = sum P[k].T*Q*P[k] + u[k].T*R*u[k]
    :param self:
    :param A:
    :param B:
    :param Q:
    :param R:
    :return:
    """
    A = eye(A.shape[0]) + ts * A
    B = ts * B
    # first, try to solve the ricatti equation
    P = solve_DARE(A, B, Q, R)
    # compute the LQR gain
    K = linalg.pinv(B.T @ P @ B + R) @ B.T @ P @ A
    return K, P

