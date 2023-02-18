#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch as t

from Comman import MemoryStore
from Evn.Quadrotor import QuadrotorFlyGui as Qgui, QuadrotorFlyModel as Qfm
from Algorithm.LQR_control import *

D2R = Qfm.D2R

class LQRControl(object):
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

        self. A = np.eye(6)
        self.A[0, 3] = self.uav_par.ts
        self.A[1, 4] = self.uav_par.ts
        self.A[2, 5] = self.uav_par.ts

        self.B = np.vstack((np.zeros((3, 3)), np.eye(3)*self.uav_par.ts))

        self.Q = np.zeros((6, 6))
        self.Q[0, 0] = 1
        self.Q[1, 1] = 1
        self.Q[2, 2] = 1
        self.Q[3, 3] = 1
        self.Q[4, 4] = 1
        self.Q[5, 5] = 1

        self.R = np.zeros((3, 3))
        self.R[0, 0] = 1
        self.R[1, 1] = 1
        self.R[2, 2] = 1

        self.k = np.array([[0.9917, 0.0000, 0.0000,  1.6693, 0.0000, -0.0000],
                  [-0.0000,  0.9917,  -0.0000, -0.0000,   1.6693, -0.0000],
                  [-0.0000, 0.0000, 0.9917, -0.0000, 0.0000,  1.6693]])
        self.err = np.zeros(12)
        print(self.A, self.B, self.k)

    def lqr_control(self, state, ref_state,
                    ref_v=np.array([0, 0, 0]),
                    ref_a=np.array([0, 0, 0])):
        """
        
        :param state: 
        :param ref_state: 
        :param ref_v: 
        :param ref_a: 
        :return: 
        """
        print('________________________________step%d simulation____________________________' % self.step_num)
        action = np.zeros(4)

        " _______________position double loop_______________ "
        # ########position loop######## #
        pos = state[0:3]
        ref_pos = ref_state[0:3]

        ref_pv = np.hstack((ref_pos, ref_v))
        err_p_pos_o = ref_pos - pos  # get new error of pos
        err_pv = state[0: 6] - ref_pv

        a_pos = - self.k @ (np.array([err_pv]).T)
        a_pos = a_pos.T[0]
        a_pos[2] += self.uav_par.g                              # gravity compensation in z-axis
        a_pos[2] = max(0.001, a_pos[2])
        a_pos += ref_a
        print(a_pos)

        " ________________attitude double loop_______________ "
        # ########attitude loop######## #
        phi = state[6]
        theta = state[7]
        phy = state[8]
        att = np.array([phi, theta, phy])


        u1 = self.uav_par.uavM * np.sqrt(sum(np.square(a_pos)))

        ref_phy = ref_state[3]
        ref_phi = np.arcsin(self.uav_par.uavM * (a_pos[0] * np.sin(ref_phy) - a_pos[1] * np.cos(ref_phy)) / u1)

        ref_theta = np.arcsin(
            self.uav_par.uavM * (a_pos[0] * np.cos(ref_phy) + a_pos[1] * np.sin(ref_phy)) / (u1 * np.cos(ref_phi)))
        ref_att = np.array([ref_phi, ref_theta, ref_phy])
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

        # a_att = a_att.clip([-25, -25, -25], [25, 25, 25])
        a_att = np.array(20 * t.tanh(t.tensor(a_att / 20)))

        u = a_att * self.uav_par.uavInertia
        u1 = max(u1, np.sqrt(np.linalg.norm(u)))
        action = np.array([u1, u[0], u[1], u[2]])
        self.err = np.array(np.hstack((err_p_pos_o, self.err_p_att, self.err_p_att_v)))
        self.step_num += 1

        return action

    def reset(self):
        self.step_num = 0
        " simulation state "
        self.err_p_att = np.zeros(3)
        self.err_i_att = np.zeros(3)
        self.err_d_att = np.zeros(3)

        self.err_p_att_v = np.zeros(3)
        self.err_i_att_v = np.zeros(3)
        self.err_d_att_v = np.zeros(3)

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

    # init controller
    controller = LQRControl(uav_para=uav_para,
                     kp_att=np.array([3.7, 3.7, 3.7]),
                     ki_att=np.array([0., 0, 0]),
                     kd_att=np.array([0, 0, 0]),
                     kp_att_v=np.array([15, 15, 15]),
                     ki_att_v=np.array([0.01, 0.01, 0.01]),
                     kd_att_v=np.array([0., 0., 0.]))

    # simulator init
    step_num = 0
    ref = np.array([10, 10, -10, 0])
    print('observe---------------------------\n', quad.observe(), '\n---------------------------observe')
    print('control---------------------------\n', controller.k, '\n---------------------------control')
    # simulate begin
    for i in range(1000):
        state_temp = quad.observe()
        action = controller.lqr_control(state_temp, ref)
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
    ts = np.array(t) * controller.ts
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
    point_track()