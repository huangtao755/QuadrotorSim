"""
introduction
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from Algorithm.ClassicControl import PidControl
from Algorithm.Velocity_Control import VelocityControl
from Comman import MemoryStore
from Evn.Quadrotor import QuadrotorFlyGui as Qgui, QuadrotorFlyModel as Qfm

D2R = Qfm.D2R


def AttitudeControl():
    """

    :return:
    """
    print("PID controller test")
    uav_para = Qfm.QuadParas(structure_type=Qfm.StructureType.quad_x)
    sim_para = Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed, init_att=np.array([0, 0, 0]),
                              init_pos=np.array([0, -10, 0]))
    quad = Qfm.QuadModel(uav_para, sim_para)
    record = MemoryStore.DataRecord()
    record.clear()

    # gui init
    gui = Qgui.QuadrotorFlyGui([quad])

    # init controller

    pid = PidControl(uav_para=uav_para,
                     kp_pos=np.array([0.5, 0.5, 0.43]),
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

    v_pid = VelocityControl(uav_para=uav_para,
                           kp_vel=np.array([1.5, 1.5, 1.4]),
                           ki_vel=np.array([0.01, 0.01, 0.01]),
                           kd_vel=np.array([0.2, 0.2, 0.]),

                           kp_att=np.array([2., 2., 2.]),
                           ki_att=np.array([0., 0, 0]),
                           kd_att=np.array([0, 0, 0]),
                           kp_att_v=np.array([15, 15, 10]),
                           ki_att_v=np.array([0.01, 0.01, 0.01]),
                           kd_att_v=np.array([0., 0., 0.01]))

    # simulator init
    step_num = 0
    ref = np.array([10, 10, -10, 0])
    print(quad.observe(), 'observe')
    # simulate begin
    gui.render()
    time.sleep(20)
    # 水平速度控制
    for i in range(400):
        print('========================================================================')
        ref_v = [0, 5, 0]
        state_temp = quad.observe()
        action = v_pid.pid_control(state=state_temp, ref_vel=ref_v)
        quad.step(action)
        if i % 2 == 0:
            gui.quadGui.target = ref[0:3]
            gui.quadGui.sim_time = quad.ts
            gui.render()
        record.buffer_append((state_temp, action))
        step_num += 1
        if state_temp[4] >= 5:
            break

    # 姿态控制
    for i in range(100):
        state_temp = quad.observe()
        kp = 3
        kd = 0.5
        u = kp * (0 - state_temp[6]) + kd * (-state_temp[9])
        action = [9.8 * uav_para.uavM + abs(u), u, 0, 0]
        quad.step(action)
        if i % 2 == 0:
            gui.quadGui.target = ref[0:3]
            gui.quadGui.sim_time = quad.ts
            gui.render()
        record.buffer_append((state_temp, action))
        step_num += 1

    # 垂直速度控制
    for i in range(200):
        state_temp = quad.observe()
        # action = pid.pid_control(state_temp, ref)
        action = [(30+9.8) * uav_para.uavM, 0, 0, 0]
        quad.step(action)
        if i % 2 == 0:
            gui.quadGui.target = ref[0:3]
            gui.quadGui.sim_time = quad.ts
            gui.render()
        record.buffer_append((state_temp, action))
        step_num += 1
        if state_temp[5] >= 8:
            break

    # 全速转体
    for i in range(15):
        state_temp = quad.observe()
        # action = pid.pid_control(state_temp, ref)
        action = [abs(np.pi/3), 0, -np.pi/3, 0]
        quad.step(action)
        if i % 1 == 0:
            gui.quadGui.target = ref[0:3]
            gui.quadGui.sim_time = quad.ts
            gui.render()
        record.buffer_append((state_temp, action))
        step_num += 1

    # pid转体
    for i in range(80):
        state_temp = quad.observe()
        # action = pid.pid_control(state_temp, ref)
        action = [0, 0, 0, 0]

        if (-100 * D2R) < state_temp[7] < (-60 * D2R):
            kp = 2
            kd = 0.5
            u = kp * (-90 * D2R - state_temp[7]) + kd * (-state_temp[10])
            u = 25 * torch.tanh(torch.tensor(u/uav_para.uavM/25)) * uav_para.uavM
            u = max(u, 0)

            action = [abs(u), 0, u, 0]
            print(action)
        quad.step(action)
        if i % 1 == 0:
            gui.quadGui.target = ref[0:3]
            gui.quadGui.sim_time = quad.ts
            gui.render()
        record.buffer_append((state_temp, action))
        step_num += 1

    # 穿越
    for i in range(20):
        state_temp = quad.observe()
        action = [0, 0, 0, 0]
        quad.step(action)
        if i % 1 == 0:
            gui.quadGui.target = ref[0:3]
            gui.quadGui.sim_time = quad.ts
            gui.render()
        record.buffer_append((state_temp, action))
        step_num += 1

    for i in range(15):
        state_temp = quad.observe()
        # action = pid.pid_control(state_temp, ref)
        action = [abs(np.pi/3), 0, np.pi/3, 0]
        quad.step(action)
        if i % 1 == 0:
            gui.quadGui.target = ref[0:3]
            gui.quadGui.sim_time = quad.ts
            gui.render()
        record.buffer_append((state_temp, action))
        step_num += 1

    for i in range(30):
        state_temp = quad.observe()
        # action = pid.pid_control(state_temp, ref)
        kp = 2
        kd = 0.4
        u = kp * (-state_temp[7]) + kd * (-state_temp[10])
        print(u, 'u')
        u = 15 * torch.tanh(torch.tensor(abs(u)/uav_para.uavM/15)) * uav_para.uavM*abs(u)/u

        action = [abs(u), 0, u, 0]
        print(action, 'actiopn')

        quad.step(action)
        if i % 1 == 0:
            gui.quadGui.target = ref[0:3]
            gui.quadGui.sim_time = quad.ts
            gui.render()
        record.buffer_append((state_temp, action))
        step_num += 1

    for i in range(100):
        print('========================================================================')
        ref_v = [0, 5, max(0, 10 - i * 0.12)]
        state_temp = quad.observe()
        action = v_pid.pid_control(state=state_temp, ref_vel=ref_v)
        quad.step(action)
        if i % 2 == 0:
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

    fig2 = plt.figure(3)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(ts, bs[t, 3], label='vx')
    plt.plot(ts, bs[t, 4], label='vy')
    plt.plot(ts, bs[t, 5], label='vz')
    plt.ylabel('v \m/s$', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.subplot(2, 1, 2)
    plt.plot(ts, bs[t, 9] / D2R, label='w_roll')
    plt.plot(ts, bs[t, 10] / D2R, label='w_pitch')
    plt.plot(ts, bs[t, 11] / D2R, label='w_yaw')
    plt.ylabel('Attitude $(\circ)$\s', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.show()


if __name__ == '__main__':
    AttitudeControl()