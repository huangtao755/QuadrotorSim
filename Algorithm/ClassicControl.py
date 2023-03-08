#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch as t

import Evn.Quadrotor.QuadrotorFlyModel as Qfm

D2R = np.pi / 180

class PidControl(object):
    def __init__(self,
                 uav_para=Qfm.QuadParas(structure_type=Qfm.StructureType.quad_x),
                 kp_pos=np.array([0, 0, 0]),
                 ki_pos=np.array([0, 0, 0]),
                 kd_pos=np.array([0, 0, 0]),
                 kp_vel=np.array([0, 0, 0]),
                 ki_vel=np.array([0, 0, 0]),
                 kd_vel=np.array([0, 0, 0]),
                 kp_att=np.array([0, 0, 0]),
                 ki_att=np.array([0, 0, 0]),
                 kd_att=np.array([0, 0, 0]),
                 kp_att_v=np.array([0, 0, 0]),
                 ki_att_v=np.array([0, 0, 0]),
                 kd_att_v=np.array([0, 0, 0])):
        """

        :param uav_para:
        :param kp_pos:
        :param ki_pos:
        :param kd_pos:
        :param kp_vel:
        :param ki_vel:
        :param kd_vel:
        :param kp_att:
        :param ki_att:
        :param kd_att:
        :param kp_att_v:
        :param ki_att_v:
        :param kd_att_v:
        """
        " init model "
        self.uav_par = uav_para
        self.ts = uav_para.ts
        self.step_num = 0
        " init control para "
        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kd_pos = kd_pos

        self.kp_vel = kp_vel
        self.ki_vel = ki_vel
        self.kd_vel = kd_vel

        self.kp_att = kp_att
        self.ki_att = ki_att
        self.kd_att = kd_att

        self.kp_att_v = kp_att_v
        self.ki_att_v = ki_att_v
        self.kd_att_v = kd_att_v
        " simulation state "
        self.err_p_pos = np.zeros(3)
        self.err_i_pos = np.zeros(3)
        self.err_d_pos = np.zeros(3)

        self.err_p_vel = np.zeros(3)
        self.err_i_vel = np.zeros(3)
        self.err_d_vel = np.zeros(3)

        self.err_p_att = np.zeros(3)
        self.err_i_att = np.zeros(3)
        self.err_d_att = np.zeros(3)

        self.err_p_att_v = np.zeros(3)
        self.err_i_att_v = np.zeros(3)
        self.err_d_att_v = np.zeros(3)

        self.err = np.zeros(12)

    def pid_control(self, state, ref_state,  compensate=np.array([0, 0, 0])):
        """

        :param state:
        :param ref_state:
        :return:
        """
        print('________________________________step%d simulation____________________________' % self.step_num)
        action = np.zeros(4)

        " _______________position double loop_______________ "
        # ########position loop######## #
        pos = state[0:3]
        ref_pos = ref_state[0:3]
        err_p_pos_o = ref_pos - pos  # get new error of pos
        err_p_pos_ = np.array(3 * t.tanh(t.tensor(err_p_pos_o / 3)))
        # err_p_pos_ = err_p_pos_.clip(np.array([-8, -8, -8]), np.array([8, 8, 8]))

        if self.step_num == 0:
            self.err_d_pos = np.zeros(3)
        else:
            self.err_d_pos = (err_p_pos_ - self.err_p_pos) / self.ts  # get new error of pos-dot
        self.err_p_pos = err_p_pos_  # update pos error
        self.err_i_pos += self.err_p_pos * self.ts  # update pos integral

        ref_vel = self.kp_pos * self.err_p_pos \
                  + self.ki_pos * self.err_i_pos \
                  + self.kd_pos * self.err_d_pos  # get ref_v as input of velocity input

        # ########velocity loop######## #
        vel = state[3:6]
        err_p_vel_ = ref_vel - vel  # get new error of velocity
        if self.step_num == 0:
            self.err_d_vel = np.zeros(3)
        else:
            self.err_d_vel = (err_p_vel_ - self.err_p_vel) / self.ts  # get new error of vel-dot
        self.err_p_vel = err_p_vel_  # update vel error
        self.err_i_vel += self.err_p_vel * self.ts  # update vel integral

        a_pos = self.kp_vel * self.err_p_vel \
                + self.ki_vel * self.err_i_vel \
                + self.kd_vel * self.err_d_vel  # get the output u of 3D for position loop

        # a_pos = a_pos.clip(np.array([-30, -30, -30]), np.array([30, 30, 30]))
        a_pos[2] += self.uav_par.g  # gravity compensation in z-axis
        a_pos[2] = max(0.001, a_pos[2])

        a_pos += compensate
        " ________________attitude double loop_______________ "
        # ########attitude loop######## #
        phi = state[6]
        theta = state[7]
        phy = state[8]
        att = np.array([phi, theta, phy])

        # u1 = self.uav_par.uavM * a_pos[2] / (np.cos(phi) * np.cos(theta))
        # print('original_u1', u1)
        u1 = self.uav_par.uavM * np.sqrt(sum(np.square(a_pos)))
        u1 = np.array(50 * t.tanh(t.tensor(u1 / 50)))

        # print('----------------------------------')
        # print('phi', phi)
        # print('theta', theta)
        # print('phy', phy)
        # print('__________________________________')

        ref_phy = ref_state[3]
        ref_phi = np.arcsin(self.uav_par.uavM * (a_pos[0] * np.sin(ref_phy) - a_pos[1] * np.cos(ref_phy)) / u1)

        ref_theta = np.arcsin(
            self.uav_par.uavM * (a_pos[0] * np.cos(ref_phy) + a_pos[1] * np.sin(ref_phy)) / (u1 * np.cos(ref_phi)))
        ref_att = np.array([ref_phi, ref_theta, ref_phy])

        # print('----------------------------------')
        # print('ref_phi', ref_phi)
        # print('ref_theta', ref_theta)
        # print('ref_phy', ref_phy)
        # print('__________________________________')

        err_p_att_ = ref_att - att

        err_p_att_[0:2] = 14 * D2R * np.tanh(err_p_att_[0:2] / D2R / 14)
        err_p_att_[2] = 10 * D2R * np.tanh(err_p_att_[2] / D2R / 10)

        if self.step_num == 0:
            self.err_d_att = np.zeros(3)
        else:
            self.err_d_att = (err_p_att_ - self.err_p_att) / self.ts
        self.err_p_att = err_p_att_
        self.err_i_att += self.err_p_att * self.ts

        ref_att_v = self.kp_att * self.err_p_att \
                    + self.ki_att * self.err_i_att \
                    + self.kd_att * self.err_d_att

        # print('----------------------------------')
        # print('err_p_att', self.err_p_att)
        # print('err_i_att', self.err_i_att)
        # print('err_d_att', self.err_d_att)
        # print('ref_att_v', ref_att_v)
        # print('__________________________________')
        # ########velocity of attitude loop######## #
        att_v = state[9:12]
        err_p_att_v_ = ref_att_v - att_v

        if self.step_num == 0:
            self.err_d_att_v = 0
        else:
            self.err_d_att_v = (err_p_att_v_ - self.err_p_att_v) / self.ts
        self.err_p_att_v = err_p_att_v_
        self.err_i_att_v += self.err_p_att_v * self.ts

        a_att = self.kp_att_v * self.err_p_att_v \
                + self.ki_att_v * self.err_i_att_v \
                + self.kd_att_v * self.err_d_att_v

        # a_att = a_att.clip([-25, -25, -25], [25, 25, 25])
        a_att = np.array(50 * t.tanh(t.tensor(a_att / 50)))

        u = a_att * self.uav_par.uavInertia
        u1 = max(u1, np.sqrt(np.linalg.norm(u)))
        action = np.array([u1, u[0], u[1], u[2]])
        self.err = np.array(np.hstack((err_p_pos_o, self.err_p_vel, self.err_p_att, self.err_p_att_v)))
        self.step_num += 1

        return action

    def attitude_control(self, state, ref_att, g_compensate=False):
        """

        :param state:
        :param ref_attitude:
        :param g_compensate:
        :return:
        """
        # ########attitude loop######## #
        phi = state[6]
        theta = state[7]
        phy = state[8]
        att = np.array([phi, theta, phy])
        a_pos = np.array([0, 0, 0])
        if g_compensate:
            a_pos[2] = self.uav_par.g
        u1 = self.uav_par.uavM * np.sqrt(sum(np.square(a_pos)))

        ref_phy = ref_att[2]
        ref_phi = ref_att[0]
        ref_theta = ref_att[1]
        err_p_att_ = ref_att - att

        if self.step_num == 0:
            self.err_d_att = np.zeros(3)
        else:
            self.err_d_att = (err_p_att_ - self.err_p_att) / self.ts
        self.err_p_att = err_p_att_
        self.err_i_att += self.err_p_att * self.ts

        ref_att_v = self.kp_att * self.err_p_att \
                    + self.ki_att * self.err_i_att \
                    + self.kd_att * self.err_d_att

        # ########velocity of attitude loop######## #
        att_v = state[9:12]
        err_p_att_v_ = ref_att_v - att_v

        if self.step_num == 0:
            self.err_d_att_v = 0
        else:
            self.err_d_att_v = (err_p_att_v_ - self.err_p_att_v) / self.ts
        self.err_p_att_v = err_p_att_v_
        self.err_i_att_v += self.err_p_att_v * self.ts

        a_att = self.kp_att_v * self.err_p_att_v \
                + self.ki_att_v * self.err_i_att_v \
                + self.kd_att_v * self.err_d_att_v

        a_att = np.array(50 * t.tanh(t.tensor(a_att / 50)))

        u = a_att * self.uav_par.uavInertia
        # u1 = max(u1, np.sqrt(np.linalg.norm(u)))
        action = np.array([u1, u[0], u[1], u[2]])
        self.step_num += 1

        return action

    def v_control(self, state):
        """

        :param state:
        :param ref_state:
        :return:
        """
        print('________________________________step%d simulation____________________________' % self.step_num)
        action = np.zeros(4)

        # ########velocity loop######## #
        ref_vel = np.array([0, 0, 0])
        vel = state[3:6]
        err_p_vel_ = ref_vel - vel  # get new error of velocity
        if self.step_num == 0:
            self.err_d_vel = np.zeros(3)
        else:
            self.err_d_vel = (err_p_vel_ - self.err_p_vel) / self.ts  # get new error of vel-dot
        self.err_p_vel = err_p_vel_  # update vel error
        self.err_i_vel += self.err_p_vel * self.ts  # update vel integral

        a_pos = self.kp_vel * self.err_p_vel \
                + self.ki_vel * self.err_i_vel \
                + self.kd_vel * self.err_d_vel  # get the output u of 3D for position loop

        # a_pos = a_pos.clip(np.array([-30, -30, -30]), np.array([30, 30, 30]))
        a_pos[2] += self.uav_par.g  # gravity compensation in z-axis
        a_pos[2] = max(0.001, a_pos[2])

        " ________________attitude double loop_______________ "
        # ########attitude loop######## #
        phi = state[6]
        theta = state[7]
        phy = state[8]
        att = np.array([phi, theta, phy])

        # u1 = self.uav_par.uavM * a_pos[2] / (np.cos(phi) * np.cos(theta))
        # print('original_u1', u1)
        u1 = self.uav_par.uavM * np.sqrt(sum(np.square(a_pos)))
        u1 = np.array(50 * t.tanh(t.tensor(u1 / 50)))

        # print('----------------------------------')
        # print('phi', phi)
        # print('theta', theta)
        # print('phy', phy)
        # print('__________________________________')

        ref_phy = 0
        ref_phi = np.arcsin(self.uav_par.uavM * (a_pos[0] * np.sin(ref_phy) - a_pos[1] * np.cos(ref_phy)) / u1)

        ref_theta = np.arcsin(
            self.uav_par.uavM * (a_pos[0] * np.cos(ref_phy) + a_pos[1] * np.sin(ref_phy)) / (u1 * np.cos(ref_phi)))
        ref_att = np.array([ref_phi, ref_theta, ref_phy])

        # print('----------------------------------')
        # print('ref_phi', ref_phi)
        # print('ref_theta', ref_theta)
        # print('ref_phy', ref_phy)
        # print('__________________________________')

        err_p_att_ = ref_att - att

        if self.step_num == 0:
            self.err_d_att = np.zeros(3)
        else:
            self.err_d_att = (err_p_att_ - self.err_p_att) / self.ts
        self.err_p_att = err_p_att_
        self.err_i_att += self.err_p_att * self.ts

        ref_att_v = self.kp_att * self.err_p_att \
                    + self.ki_att * self.err_i_att \
                    + self.kd_att * self.err_d_att

        # print('----------------------------------')
        # print('err_p_att', self.err_p_att)
        # print('err_i_att', self.err_i_att)
        # print('err_d_att', self.err_d_att)
        # print('ref_att_v', ref_att_v)
        # print('__________________________________')
        # ########velocity of attitude loop######## #
        att_v = state[9:12]
        err_p_att_v_ = ref_att_v - att_v

        if self.step_num == 0:
            self.err_d_att_v = 0
        else:
            self.err_d_att_v = (err_p_att_v_ - self.err_p_att_v) / self.ts
        self.err_p_att_v = err_p_att_v_
        self.err_i_att_v += self.err_p_att_v * self.ts

        a_att = self.kp_att_v * self.err_p_att_v \
                + self.ki_att_v * self.err_i_att_v \
                + self.kd_att_v * self.err_d_att_v

        # a_att = a_att.clip([-25, -25, -25], [25, 25, 25])
        a_att = np.array(50 * t.tanh(t.tensor(a_att / 50)))

        u = a_att * self.uav_par.uavInertia
        u1 = max(u1, np.sqrt(np.linalg.norm(u)))
        action = np.array([u1, u[0], u[1], u[2]])
        # self.err = np.array(np.hstack((err_p_pos_o, self.err_p_vel, self.err_p_att, self.err_p_att_v)))
        self.step_num += 1

        return action

    def reset(self):
        self.step_num = 0
        " simulation state "
        self.err_p_pos = np.zeros(3)
        self.err_i_pos = np.zeros(3)
        self.err_d_pos = np.zeros(3)

        self.err_p_vel = np.zeros(3)
        self.err_i_vel = np.zeros(3)
        self.err_d_vel = np.zeros(3)

        self.err_p_att = np.zeros(3)
        self.err_i_att = np.zeros(3)
        self.err_d_att = np.zeros(3)

        self.err_p_att_v = np.zeros(3)
        self.err_i_att_v = np.zeros(3)
        self.err_d_att_v = np.zeros(3)


""""""
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
