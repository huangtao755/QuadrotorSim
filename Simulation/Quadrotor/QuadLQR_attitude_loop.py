#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch as t

from Algorithm.LQR_control import *
from Evn.Quadrotor import QuadrotorFlyModel
from Comman import MemoryStore
from Evn.Quadrotor import QuadrotorFlyGui as Qgui, QuadrotorFlyModel as Qfm

D2R = Qfm.D2R

class PidControl(object):
    def __init__(self,
                 uav_para=Qfm.QuadParas(structure_type=Qfm.StructureType.quad_x),
                 kp_att=np.array([0, 0, 0]),
                 ki_att=np.array([0, 0, 0]),
                 kd_att=np.array([0, 0, 0]),
                 kp_att_v=np.array([0, 0, 0]),
                 ki_att_v=np.array([0, 0, 0]),
                 kd_att_v=np.array([0, 0, 0])):
        """

        :param uav_para:
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

        self.kp_att = kp_att
        self.ki_att = ki_att
        self.kd_att = kd_att

        self.kp_att_v = kp_att_v
        self.ki_att_v = ki_att_v
        self.kd_att_v = kd_att_v
        " simulation state "

        self.err_p_att = np.zeros(3)
        self.err_i_att = np.zeros(3)
        self.err_d_att = np.zeros(3)

        self.err_p_att_v = np.zeros(3)
        self.err_i_att_v = np.zeros(3)
        self.err_d_att_v = np.zeros(3)

        self.err = np.zeros(12)

        self.A = None
        self.B = None
        self.angle_matrix(p=0, q=0, r=0)

        Q = np.zeros((6, 6))
        Q[0, 0] = 1000
        Q[1, 1] = 1000
        Q[2, 2] = 1000
        Q[3, 3] = 10
        Q[4, 4] = 10
        Q[5, 5] = 10

        R = np.zeros((3, 3))
        R[0, 0] = 1
        R[1, 1] = 1
        R[2, 2] = 1
        self.Q = Q
        self.R = R

    def LQR_control(self, state, ref_state,
                    compensate=np.array([0, 0, 0])):
        """

        :param state:
        :param ref_state:
        :param compensate:
        :return:
        """
        print('________________________________step%d simulation____________________________' % self.step_num)
        action = np.zeros(4)

        " _______________position double loop_______________ "
        # ########position loop######## #
        pos = state[0:3]
        ref_pos = ref_state[0:3]
        err_p_pos_o = ref_pos - pos  # get new error of pos
        err_p_pos_ = np.array(8 * t.tanh(t.tensor(err_p_pos_o / 8)))
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

        u1 = self.uav_par.uavM * \
             np.sqrt(sum(np.square(a_pos)))

        ref_phy = ref_state[3]
        ref_phi = np.arcsin(self.uav_par.uavM * (a_pos[0] * np.sin(ref_phy) - a_pos[1] * np.cos(ref_phy)) / u1)

        ref_theta = np.arcsin(
            self.uav_par.uavM * (a_pos[0] * np.cos(ref_phy) + a_pos[1] * np.sin(ref_phy)) / (u1 * np.cos(ref_phi)))
        ref_att = np.array([ref_phi, ref_theta, ref_phy])

        self.A, self.B = self.angle_matrix(p=phi, q=theta, r=phy)

        K, P = lqr(self.A, self.B, self.Q, self.R, ts=0.01)

        err_p_att_ = ref_att - att

        err_v_att = state[9:]

        feedback_state = np.hstack((-err_p_att_, err_v_att))

        a_att = - K @ feedback_state.T

        # if self.step_num == 0:
        #     self.err_d_att = np.zeros(3)
        # else:
        #     self.err_d_att = (err_p_att_ - self.err_p_att) / self.ts
        # self.err_p_att = err_p_att_
        # self.err_i_att += self.err_p_att * self.ts
        #
        # ref_att_v = self.kp_att * self.err_p_att \
        #             + self.ki_att * self.err_i_att \
        #             + self.kd_att * self.err_d_att
        #
        # # ########velocity of attitude loop######## #
        # att_v = state[9:12]
        # err_p_att_v_ = ref_att_v - att_v
        #
        # if self.step_num == 0:
        #     self.err_d_att_v = 0
        # else:
        #     self.err_d_att_v = (err_p_att_v_ - self.err_p_att_v) / self.ts
        # self.err_p_att_v = err_p_att_v_
        # self.err_i_att_v += self.err_p_att_v * self.ts
        #
        # a_att = self.kp_att_v * self.err_p_att_v \
        #         + self.ki_att_v * self.err_i_att_v \
        #         + self.kd_att_v * self.err_d_att_v
        #
        # # a_att = a_att.clip([-25, -25, -25], [25, 25, 25])
        # a_att = np.array(20 * t.tanh(t.tensor(a_att / 20)))

        u = a_att * self.uav_par.uavInertia
        u1 = max(u1, np.sqrt(np.linalg.norm(u)))
        action = np.array([u1, u[0], u[1], u[2]])
        self.err = np.array(np.hstack((err_p_pos_o, self.err_p_vel, err_p_att_)))
        self.step_num += 1

        return action

    def angle_matrix(self, p, q, r):
        A_22 = np.array([[0,
                         (self.uav_par.uavInertia[1] - self.uav_par.uavInertia[2]) * r / self.uav_par.uavInertia[0],
                         (self.uav_par.uavInertia[1] - self.uav_par.uavInertia[2]) * q / self.uav_par.uavInertia[0]],
                        [(self.uav_par.uavInertia[2] - self.uav_par.uavInertia[0]) * r / self.uav_par.uavInertia[1],
                         0,
                         (self.uav_par.uavInertia[2] - self.uav_par.uavInertia[0]) * p / self.uav_par.uavInertia[1]],
                        [(self.uav_par.uavInertia[0] - self.uav_par.uavInertia[1]) * q / self.uav_par.uavInertia[2],
                         (self.uav_par.uavInertia[0] - self.uav_par.uavInertia[1]) * p / self.uav_par.uavInertia[2],
                         0]])

        A = np.array(np.vstack((np.hstack((np.eye(3), np.zeros((3, 3)))),
                               np.hstack((np.zeros((3, 3)), A_22)))))
        print(A, 'A')

        J = np.array([[self.uav_par.uavInertia[0], 0, 0],
                      [0, self.uav_par.uavInertia[1], 0],
                      [0, 0, self.uav_par.uavInertia[2]]])

        print(J, 'j')
        B = np.vstack((np.zeros((3, 3)), J))
        print(B, 'B')
        self.A, self.B = A, B
        return A, B

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


pid = PidControl()
A, B = pid.angle_matrix(p=0.0, q=0.0, r=0)

Q = np.zeros((6, 6))
Q[0, 0] = 1000
Q[1, 1] = 1000
Q[2, 2] = 1000
Q[3, 3] = 10
Q[4, 4] = 10
Q[5, 5] = 10

R = np.zeros((3, 3))
R[0, 0] = 1
R[1, 1] = 1
R[2, 2] = 1

K, P = lqr(A, B, Q, R, ts=0.01)
print(K)

def point_track():
    """

    :return:
    """
    print("PID controller test")
    uav_para = Qfm.QuadParas(structure_type=Qfm.StructureType.quad_x)
    sim_para = Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed, init_att=np.array([0, 0, 0]),
                              init_pos=np.array([0, 0, 0]))
    quad = Qfm.QuadModel(uav_para, sim_para)
    record = MemoryStore.DataRecord()
    record.clear()

    # gui init
    gui = Qgui.QuadrotorFlyGui([quad])
    pid = PidControl(uav_para=uav_para,
                     kp_pos=np.array([0.5, 0.5, 0.5]),
                     ki_pos=np.array([0, 0., 0.0]),
                     kd_pos=np.array([0, 0, 0]),
                     kp_vel=np.array([1.5, 1.5, 1.4]),
                     ki_vel=np.array([0.01, 0.01, 0.01]),
                     kd_vel=np.array([0.1, 0.1, 0.]),

                     kp_att=np.array([2., 2., 2.]),
                     ki_att=np.array([0., 0, 0]),
                     kd_att=np.array([0, 0, 0]),
                     kp_att_v=np.array([12, 12, 10]),
                     ki_att_v=np.array([0.01, 0.01, 0.01]),
                     kd_att_v=np.array([0., 0., 0.01]))

    # simulator init
    step_num = 0
    ref = np.array([10, 10, -10, 0])
    print(quad.observe(), 'observe')
    # simulate begin
    for i in range(1300):
        state_temp = quad.observe()
        action = pid.LQR_control(state_temp, ref)
        quad.step(action)
        if i % 10 == 0:
            gui.quadGui.target = ref[0:3]
            gui.quadGui.sim_time = quad.ts
            gui.render()
        record.buffer_append((state_temp, action))
        step_num += 1

    record.episode_append()
    data = record.get_episode_buffer()
    bs = data[0]
    ba = data[1]
    t = range(0, record.count)
    ts = np.array(t) * pid.ts
    # mpl.style.use('seaborn')
    fig1 = plt.figure(2)
    plt.clf()
    plt.subplot(4, 1, 1)
    plt.plot(ts, bs[t, 6] / D2R, label='roll')
    plt.plot(ts, bs[t, 7] / D2R, label='pitch')
    plt.plot(ts, bs[t, 8] / D2R, label='yaw')
    plt.ylabel('Attitude $(\circ)$', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.subplot(4, 1, 2)
    plt.plot(ts, bs[t, 0], label='x')
    plt.plot(ts, bs[t, 1], label='y')
    plt.ylabel('Position (m)', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.subplot(4, 1, 3)
    plt.plot(ts, bs[t, 2], label='z')
    plt.ylabel('Altitude (m)', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.subplot(4, 1, 4)
    plt.plot(ts, ba[t, 0], label='f')
    plt.plot(ts, ba[t, 1] / uav_para.uavInertia[0], label='t1')
    plt.plot(ts, ba[t, 2] / uav_para.uavInertia[1], label='t2')
    plt.plot(ts, ba[t, 3] / uav_para.uavInertia[2], label='t3')
    plt.ylabel('f (m/s^2)', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.show()

if __name__ == "__main__":
    pid = PidControl()
    A, B = pid.angle_matrix(p=1, q=1, r=1)

    Q = np.zeros((6, 6))
    Q[0, 0] = 1000
    Q[1, 1] = 1000
    Q[2, 2] = 1000
    Q[3, 3] = 10
    Q[4, 4] = 10
    Q[5, 5] = 10
    print(Q)
    R = np.zeros((3, 3))
    R[0, 0] = 1
    R[1, 1] = 1
    R[2, 2] = 1
    print(R)
    K = lqr(A, B, Q, R)
    print(K)