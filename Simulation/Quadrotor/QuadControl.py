#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
introduction:
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from Algorithm.ClassicControl import PidControl
from Comman import MemoryStore
from Evn.Quadrotor import QuadrotorFlyGui as Qgui, QuadrotorFlyModel as Qfm

D2R = Qfm.D2R
current_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))


class QuadControl(object):
    """
    :brief:
    """
    def __init__(self,
                 structure_type=Qfm.StructureType.quad_x,
                 init_att=np.array([0, 0, 0]),
                 init_pos=np.array([0, 0, 0]),
                 name='quad'):
        """

        :param structure_type:
        :param init_att:
        :param init_pos:
        """
        self.uav_para = Qfm.QuadParas(structure_type=structure_type)
        self.sim_para = Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed, init_att=init_att,
                                       init_pos=init_pos)
        self.quad = Qfm.QuadModel(self.uav_para, self.sim_para)
        # self.pid = PidControl(uav_para=self.uav_para,
        #                       kp_pos=np.array([0.52, 0.52, 0.52]),
        #                       ki_pos=np.array([0, 0., 0.0]),
        #                       kd_pos=np.array([0, 0, 0]),
        #                       kp_vel=np.array([1.6, 1.6, 1.8]),
        #                       ki_vel=np.array([0.0, 0.0, 0.0]),
        #                       kd_vel=np.array([0.05, 0.05, 0.05]),
        #
        #                       kp_att=np.array([2., 2., 2.]),
        #                       ki_att=np.array([0., 0, 0]),
        #                       kd_att=np.array([0, 0, 0]),
        #                       kp_att_v=np.array([25, 25, 10]),
        #                       ki_att_v=np.array([0.0, 0.0, 0.0]),
        #                       kd_att_v=np.array([0.05, 0.05, 0.01]))
        self.pid = PidControl(uav_para=self.uav_para,
                         kp_pos=np.array([0.6, 0.6, 0.6]),
                         ki_pos=np.array([0., 0., 0.0]),
                         kd_pos=np.array([0., 0., 0.]),
                         kp_vel=np.array([1.5, 1.5, 1.5]),
                         ki_vel=np.array([0.15, 0.15, 0.1]),
                         kd_vel=np.array([0., 0., 0.]),

                         kp_att=np.array([3, 3, 2.]),
                         ki_att=np.array([0., 0, 0]),
                         kd_att=np.array([0., 0., 0]),
                         kp_att_v=np.array([13, 13, 10]),
                         ki_att_v=np.array([0.01, 0.01, 0.01]),
                         kd_att_v=np.array([0., 0., 0.01]))

        self.state_temp = self.quad.observe()
        self.record = MemoryStore.DataRecord()
        self.record.clear()
        self.step_num = 0

        self.name = name

        self.ref = np.array([0, 0, 0, 0])
        self.ref_v = np.array([0, 0, 0, 0])
        self.track_err = None
        self.fig1 = None
        self.fig2 = None

    def track(self, steps: int, ref, ref_v=np.array([0, 0, 0, 0])):
        """

        :param steps:
        :param ref:
        :param ref_v:
        :return:
        """
        self.state_temp = self.quad.observe()
        state_compensate = self.state_temp - np.array([0, 0, 0,
                                                       ref_v[0], ref_v[1], ref_v[2],
                                                       0, 0, 0,
                                                       0, 0, ref_v[3]])
        action = self.pid.pid_control(state_compensate, ref)
        self.quad.step(action)

        track_err = np.hstack([ref[0] - self.state_temp[0],
                               ref[1] - self.state_temp[1],
                               ref[2] - self.state_temp[2],
                               ref[3] - self.state_temp[8]])
        self.track_err = track_err[0:3]
        err_state = self.pid.err
        print((self.state_temp, action, track_err, err_state), 'err_state',
              '-------------------------------------------------------------------------------------------------------------------')
        self.record.buffer_append((self.state_temp, action, track_err, err_state))

        self.step_num += 1
        print('steps_num', self.step_num)
        if self.step_num == steps:
            print('ending')
            self.record.episode_append()

    def data_save(self):
        """

        :return:
        """
        data = self.record.get_episode_buffer()
        print(len(data))
        state_data = data[0]
        action_data = data[1]
        track_err_data = data[2]

        self.record.save_data(path=current_path + '//DataSave//QuadPid', data_name=str(self.name + '_state'),
                              data=state_data)
        self.record.save_data(path=current_path + '//DataSave//QuadPid', data_name=self.name + '_action',
                              data=action_data)
        self.record.save_data(path=current_path + '//DataSave//QuadPid', data_name=self.name + '_track_err',
                              data=track_err_data)
        if len(data) == 4:
            err_state = data[3]
            self.record.save_data(path=current_path + '//DataSave//QuadPid', data_name=self.name + '_state_err',
                                  data=err_state)
        else:
            print(self.name + 'Have no state_err')

    def reset(self, att, pos):
        """

        :param att:
        :param pos:
        :return:
        """
        self.quad.reset_states(att=att, pos=pos)
        self.step_num = 0
        self.record.clear()

    def fig_show(self, i):
        """

        :param i:
        :return:
        """
        data = self.record.get_episode_buffer()
        bs = data[0]
        ba = data[1]
        t = range(0, self.record.count)
        ts = np.array(t) * self.pid.ts

        self.fig1 = plt.figure(int(i * 3))
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
        plt.ylabel('Altitude $(\circ)$', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        plt.subplot(4, 1, 4)
        plt.plot(ts, ba[t, 0], label='f')
        plt.plot(ts, ba[t, 1] / self.uav_para.uavInertia[0], label='t1')
        plt.plot(ts, ba[t, 2] / self.uav_para.uavInertia[1], label='t2')
        plt.plot(ts, ba[t, 3] / self.uav_para.uavInertia[2], label='t3')
        plt.ylabel('f (m/s^2)', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        # print(data)
        if len(data) == 4:
            err = data[2]
            self.fig1 = plt.figure(int(i * 3 + 1))
            plt.subplot(3, 1, 1)
            plt.plot(ts, bs[t, 3], label='x')
            plt.plot(ts, bs[t, 4], label='y')
            plt.plot(ts, bs[t, 5], label='z')
            plt.ylabel('v/m/s', fontsize=15)
            plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
            plt.subplot(3, 1, 2)
            plt.plot(ts, err[t, 0], label='x')
            plt.plot(ts, err[t, 1], label='y')
            plt.plot(ts, err[t, 2], label='z')
            plt.ylabel('error/m', fontsize=15)
            plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
            plt.subplot(3, 1, 3)
            plt.plot(ts, err[t, 3], label='err_phi')
            plt.ylabel('Altitude_err $(\circ)$', fontsize=15)
            plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        # plt.show()


def main():
    """

    :return:
    """
    quad1 = QuadControl(init_pos=np.array([0, 0, 0]), name='quad1')
    quad2 = QuadControl(init_pos=np.array([0, 0, 0]), name='quad2')
    gui = Qgui.QuadrotorFlyGui([quad1.quad, quad2.quad])
    steps = 1000
    ref = np.array([-15, -15, -15, 0])
    ref_v = np.array([0, 0, 0, 0])
    for i in range(steps):
        # ref = np.array([3 * np.cos(np.pi / 18 * quad1.quad.ts + np.pi),
        #                 3 * np.sin(np.pi / 18 * quad1.quad.ts + np.pi),
        #                 0.2 * quad1.quad.ts,
        #                 np.pi / 18 * quad1.quad.ts])
        # ref_v = np.array([-np.pi * np.sin(np.pi / 18 * quad1.quad.ts + np.pi) * 3 / 18,
        #                  np.pi * np.cos(np.pi / 18 * quad1.quad.ts + np.pi) * 3 / 18,
        #                  0.2,
        #                  np.pi / 18])  # target velocity

        quad2.state_temp = quad2.quad.observe()

        quad1.track(ref=ref, ref_v=ref_v, steps=steps)
        state_compensate = quad2.state_temp - np.array([0, 0, 0,
                                                        ref_v[0], ref_v[1], ref_v[2],
                                                        0, 0, 0,
                                                        0, 0, ref_v[3]])
        action2 = quad2.quad.controller_pid(state=state_compensate, ref_state=ref)
        quad2.quad.step(action2)
        track_err = np.hstack([ref[0] - quad2.state_temp[0],
                               ref[1] - quad2.state_temp[1],
                               ref[2] - quad2.state_temp[2],
                               ref[3] - quad2.state_temp[8]])

        quad2.record.buffer_append((quad2.state_temp, action2, track_err))
        sum_err = np.sqrt(np.linalg.norm(quad1.track_err[0:3], ord=2))
        # print(track_err, sum_err)
        if sum_err < 0.01:
            print(i)
            return print('reach point')

        if i % 20 == 0:
            gui.quadGui.target = ref
            gui.quadGui.sim_time = quad1.quad.ts
            gui.render()
    quad1.data_save()
    print('datasave')
    quad2.record.episode_append()
    quad2.data_save()

    quad1.fig_show(1)
    quad2.fig_show(2)
    plt.show()


if __name__ == "__main__":
    main()
