import os
import matplotlib.pyplot as plt
import numpy as np

from Algorithm.ClassicControl import PidControl
from Comman import MemoryStore
from Evn.Quadrotor import QuadrotorFlyGui as Qgui, QuadrotorFlyModel_v2 as Qfm

D2R = Qfm.D2R
current_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))


class QuadControl(Qfm.QuadModel):
    """
    :brief:
    """
    def __init__(self,
                 uav_para,
                 sim_para,
                 structure_type=Qfm.StructureType.quad_x,
                 init_att=np.array([0, 0, 0]),
                 init_pos=np.array([0, 0, 0]),
                 name='quad'):
        """

        :param structure_type:
        :param init_att:
        :param init_pos:
        """
        super(QuadControl, self).__init__(uav_para, sim_para)
        self.state_temp = self.observe()
        self.record = MemoryStore.DataRecord()
        self.record.clear()
        self.step_num = 0

        self.name = name

        self.ref = np.array([0, 0, 0, 0])
        self.ref_v = np.array([0, 0, 0, 0])
        self.track_err = None
        self.fig1 = None
        self.fig2 = None
        a = np.ones(8)*6
        b = np.ones(4)
        self.c = np.hstack((a, b))

    def back_step_control(self, ref, ref_v, ref_a, action):
        para = self.uavPara
        c = self.c

        att_cos = np.cos(self.attitude)
        att_sin = np.sin(self.attitude)

        self.rotor_mat = np.array([[1, att_sin[1]*att_sin[0]/att_cos[1], att_sin[1]*att_cos[0]/att_cos[1]],
                                       [0, att_cos[0], -att_sin[0]],
                                       [0, att_sin[0]/att_cos[1], att_cos[0]/att_cos[1]]])

        state_body = (np.linalg.inv(self.rotor_mat) @ self.angular.T).reshape(3)
        state_temp = np.hstack([self.position, self.velocity, self.attitude, state_body])
        # state [x, y, z, vx, vy, vz, phi, theta, pis, p, q, r]

        dot_state = self.dynamic_basic(self.state_temp, action)

        u1 = para.uavM / (np.cos(state_temp[6])*np.cos(state_temp[7])) *\
            (- (1 + c[8] * c[9]) * (state_temp[2] - ref[2])
             - (c[8] + c[9]) * (state_temp[5] - ref_v[2])
             + para.g + dot_state[2])

        u2 = para.uavM * para.uavInertia[0] / (u1 * para.uavL * np.cos(state_temp[6]) * np.cos(state_temp[8])) \
            * ((1 + c[4]*c[5] + c[4]*c[7] + c[6]*c[7] + c[4]*c[5]*c[6]*c[7])*(state_temp[1] - ref[1])
               - (2*c[4] + c[5] + c[6] + 2 * c[7] + c[4]*c[5]*c[6] + c[4]*c[5]*c[7] + c[4]*c[6]*c[7] + c[5]*c[6]*c[7])
               * (state_temp[4] - ref_v[1]) - (3 + c[4]*c[5] + c[5]*c[6] + c[4]*c[7] + c[5]*c[6] + c[5]*c[7] + c[6]*c[7])
               * u1/para.uavM*np.cos(state_temp[8])*np.sin(state_temp[6])
               - (c[4] + c[5] + c[6] + c[7]) * (u1 / para.uavM * state_temp[9]*np.cos(state_temp[8])*np.cos(state_temp[6])
               - u1 / para.uavM * state_temp[11]*np.sin(state_temp[8])*np.sin(state_temp[6]))
               + u1 / para.uavM * (state_temp[9] * state_temp[9] * np.cos(state_temp[8]) * np.sin(state_temp[6]))
               + u1 / para.uavM * (state_temp[11] * state_temp[11] * np.cos(state_temp[8]) * np.sin(state_temp[6]))
               + u1 / para.uavM * 2 * (state_temp[9] * state_temp[11] * np.cos(state_temp[6]) * np.sin(state_temp[8]))
               + u1 / para.uavM * 2 * (dot_state[11] * np.sin(state_temp[8]) * np.sin(state_temp[6])))\
            - (para.uavInertia[1] - para.uavInertia[2]) / para.uavL * state_temp[10] * state_temp[11]

        u3 = para.uavM * para.uavInertia[0] / (u1 * para.uavL * np.cos(state_temp[7]) * np.cos(state_temp[8])) \
            * (- (1 + c[0]*c[1] + c[0]*c[3] + c[2]*c[3] + c[0]*c[1]*c[2]*c[3])*(state_temp[0] - ref[0])
               - (2*c[0] + c[1] + c[2] + 2 * c[3] + c[0]*c[1]*c[2] + c[0]*c[1]*c[2] + c[0]*c[2]*c[3] + c[1]*c[2]*c[3])
               * (state_temp[3] - ref_v[0]) - (3 + c[0]*c[1] + c[0]*c[2] + c[0]*c[3] + c[1]*c[2] + c[1]*c[3] + c[2]*c[3])
               * u1/para.uavM*np.cos(state_temp[8])*np.sin(state_temp[7])
               - (c[0] + c[1] + c[2] + c[3]) * (u1 / para.uavM * state_temp[10]*np.cos(state_temp[8])*np.cos(state_temp[7])
               - u1 / para.uavM * state_temp[11]*np.sin(state_temp[8])*np.sin(state_temp[7]))
               + u1 / para.uavM * (state_temp[10] * state_temp[10] * np.cos(state_temp[8]) * np.sin(state_temp[7]))
               + u1 / para.uavM * (state_temp[11] * state_temp[11] * np.cos(state_temp[8]) * np.sin(state_temp[7]))
               + u1 / para.uavM * 2 * (state_temp[10] * state_temp[11] * np.cos(state_temp[7]) * np.sin(state_temp[8]))
               + u1 / para.uavM * 2 * (dot_state[11] * np.sin(state_temp[8]) * np.sin(state_temp[7])))\
            - (para.uavInertia[2] - para.uavInertia[0]) / para.uavL * state_temp[9] * state_temp[11]

        u4 = para.uavInertia[2] * (- (1 + c[10]*c[11]) * (state_temp[8] - ref[3])
                                   - (c[10]+c[11])*(state_temp[11] - ref_v[3] + para.g + dot_state[2]))\
            - (para.uavInertia[0] - para.uavInertia[1]) * state_temp[9] * state_temp[10]

        action = np.array([u1, u2, u3, u4])
        return action


def traject_track():
    print('back_step control test')

    uav_para = Qfm.QuadParas(structure_type=Qfm.StructureType.quad_x)
    sim_para = Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed,
                              init_att=np.array([0, 0, 0]), init_pos=np.array([-2, 0, 0]))
    quad = QuadControl(uav_para=uav_para,
                       sim_para=sim_para)
    record = MemoryStore.DataRecord()
    record.clear()

    # gui init
    gui = Qgui.QuadrotorFlyGui([quad])
