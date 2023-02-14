"""
introduction
"""
import matplotlib.pyplot as plt
import numpy as np

from Algorithm.ClassicControl import PidControl
from Comman import MemoryStore
from Evn.Quadrotor import QuadrotorFlyGui as Qgui, QuadrotorFlyModel as Qfm

D2R = Qfm.D2R


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
        action = pid.pid_control(state_temp, ref)
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


def traject_track():
    """

    :return:
    """
    print("PID controller test")
    uav_para = Qfm.QuadParas(structure_type=Qfm.StructureType.quad_x)
    sim_para = Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed,
                              init_att=np.array([0, 0, 0]), init_pos=np.array([-2, 0, 0]))
    quad = Qfm.QuadModel(uav_para, sim_para)
    record = MemoryStore.DataRecord()
    record.clear()

    # gui init
    gui = Qgui.QuadrotorFlyGui([quad])

    # init controller
    # pid = PidControl(uav_para=uav_para,
    #                  kp_pos=np.array([0.5, 0.5, 0.5]),
    #                  ki_pos=np.array([0, 0., 0.0]),
    #                  kd_pos=np.array([0.1, 0.1, 0.1]),
    #                  kp_vel=np.array([1.5, 1.5, 1.5]),
    #                  ki_vel=np.array([0.01, 0.01, 0.01]),
    #                  kd_vel=np.array([0., 0., 0.]),
    #
    #                  kp_att=np.array([3., 3., 3.]),
    #                  ki_att=np.array([0., 0, 0]),
    #                  kd_att=np.array([0, 0, 0]),
    #                  kp_att_v=np.array([11, 11, 10]),
    #                  ki_att_v=np.array([0.01, 0.01, 0.01]),
    #                  kd_att_v=np.array([0., 0., 0.01]))

    pid = PidControl(uav_para=uav_para,
                    kp_pos = np.array([0.8, 0.8, 0.8]),
                    ki_pos = np.array([0., 0., 0.0]),
                    kd_pos = np.array([0., 0., 0.]),
                    kp_vel = np.array([2, 2, 2]),
                    ki_vel = np.array([0.15, 0.15, 0.01]),
                    kd_vel = np.array([0., 0., 0.]),

                    kp_att = np.array([3, 3, 2.]),
                    ki_att = np.array([0., 0, 0]),
                    kd_att = np.array([0., 0., 0]),
                    kp_att_v = np.array([20, 20, 10]),
                    ki_att_v = np.array([1, 1, 1]),
                    kd_att_v = np.array([0., 0., 0.01]))

    # simulator init
    step_num = 0
    ref = np.array([15, -15, -15, 0])
    ref_v = np.array([0, 0, 0, 0])
    print(quad.observe())
    # simulate begin
    for i in range(5000):

        ref = np.array([5 * np.cos(np.pi / 5 * quad.ts + np.pi),
                        5 * np.sin(np.pi / 5 * quad.ts + np.pi),
                        0.0 * quad.ts, 0])
                        # np.pi / 12 * quad.ts])

        ref_v = np.array([-np.pi * np.sin(np.pi / 5 * (quad.ts + quad.uavPara.ts) + np.pi) * 5 / 5,
                          np.pi * np.cos(np.pi / 5 * (quad.ts + quad.uavPara.ts) + np.pi) * 5 / 5,
                          0.0,
                          0.0])
        ref_a = np.array([-np.pi**2 * np.cos(np.pi / 5 * quad.ts + np.pi) * 5 / 25,
                          -np.pi**2 * np.sin(np.pi / 5 * quad.ts + np.pi) * 5 / 25,
                          0.0])
        # ref_a = [0, 0, 0]
                          # np.pi / 12])  # target velocity
        state_temp = quad.observe()
        print(state_temp, 'state')
        state_compensate = state_temp - np.array([0, 0, 0,
                                                  ref_v[0], ref_v[1], ref_v[2],
                                                  0, 0, 0,
                                                  0, 0, ref_v[3]])
        # state_compensate = state_temp
        action = pid.pid_control(state_compensate, ref + ref_v * quad.uavPara.ts, compensate=ref_a)
        # action = quad.controller_pid(state_compensate, ref_state=ref)
        quad.step(action)

        err_track = np.hstack([ref[0] - state_temp[0],
                               ref[1] - state_temp[1],
                               ref[2] - state_temp[2],
                               ref[3] - state_temp[8]])
        if i % 20 == 0:
            gui.quadGui.target = ref[0:3]
            gui.quadGui.sim_time = quad.ts
            gui.render()
        state_temp[8] = state_temp[8] % (2 * np.pi)
        # print(state_temp[8], ''' angle''')
        record.buffer_append((state_temp, action, err_track))

        step_num += 1

    # print(track_err)
    record.episode_append()

    data = record.get_episode_buffer()
    print(data[0])
    bs = data[0]
    ba = data[1]
    err = data[2]
    print(len(data), 'length data')
    t = range(0, record.count)
    ts = np.array(t) * pid.ts
    # mpl.style.use('seaborn')
    fig1 = plt.figure(2)
    plt.clf()
    plt.subplot(5, 1, 1)
    plt.plot(ts, bs[t, 6] / D2R, label='roll')
    plt.plot(ts, bs[t, 7] / D2R, label='pitch')
    plt.plot(ts, bs[t, 8] / D2R, label='yaw')
    plt.ylabel('Attitude $(\circ)$', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.subplot(5, 1, 2)
    plt.plot(ts, bs[t, 9] / D2R, label='roll_dot')
    plt.plot(ts, bs[t, 10] / D2R, label='pitch_dot')
    plt.plot(ts, bs[t, 11] / D2R, label='yaw_dot')
    plt.ylabel('Attitude_dot $(\circ)$/s', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.subplot(5, 1, 3)
    plt.plot(ts, bs[t, 0], label='x')
    plt.plot(ts, bs[t, 1], label='y')
    plt.ylabel('Position (m)', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.subplot(5, 1, 4)
    plt.plot(ts, bs[t, 2], label='z')
    plt.ylabel('Altitude $(\circ)$', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.subplot(5, 1, 5)
    plt.plot(ts, ba[t, 0], label='f')
    plt.plot(ts, ba[t, 1] / uav_para.uavInertia[0], label='t1')
    plt.plot(ts, ba[t, 2] / uav_para.uavInertia[1], label='t2')
    plt.plot(ts, ba[t, 3] / uav_para.uavInertia[2], label='t3')
    plt.ylabel('f (m/s^2)', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))

    fig2 = plt.figure(3)
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
    plt.show()


if __name__ == "__main__":
    traject_track()
