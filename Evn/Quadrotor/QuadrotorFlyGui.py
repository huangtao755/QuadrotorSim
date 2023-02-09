#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import numpy as np

from Comman import CommonFunctions as Cf
from Comman import MemoryStore
from Evn.Quadrotor import QuadrotorFlyModel as Qfm


class QuadrotorFlyGuiEnv(object):
    def __init__(self, bound_x=20., bound_y=20., bound_z=15.):
        """Define the environment of quadrotor simulation
        :param bound_x:
        :param bound_y:
        :param bound_z:
        """
        self.fig = plt.figure(figsize=(16, 9))
        self.boundX = bound_x * 1.
        self.boundY = bound_y * 1.
        self.boundZ = bound_z * 1.
        self.ax = axes3d.Axes3D(self.fig)
        self.ax.set_xlim3d([-self.boundX, self.boundX])
        self.ax.set_ylim3d([-self.boundY, self.boundY])
        self.ax.set_zlim3d([-self.boundZ, self.boundZ])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('QuadrotorFly Simulation', fontsize='13')


# def get_rotation_matrix(att):
#     cos_att = np.cos(att)
#     sin_att = np.sin(att)
#
#     rotation_x = np.array([[1, 0, 0], [0, cos_att[0], -sin_att[0]], [0, sin_att[0], cos_att[0]]])
#     rotation_y = np.array([[cos_att[1], 0, sin_att[1]], [0, 1, 0], [-sin_att[1], 0, cos_att[1]]])
#     rotation_z = np.array([[cos_att[2], -sin_att[2], 0], [sin_att[2], cos_att[2], 0], [0, 0, 1]])
#     rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
#
#     return rotation_matrix


class QuadrotorFlyGuiUav(object):
    """Draw quadrotor class"""

    def __init__(self, quads: list, ax: axes3d.Axes3D):
        self.quads = list()
        self.quadGui = list()
        self.traject = list()
        self.target = [0, 0, 0]
        self.sim_time = 0
        self.color = ['r', 'b', 'g', 'k', 'm', 'y', 'k']
        self.ax = ax

        self.time = self.ax.text2D(0.02, 0.87, 'time/s:' + str(self.sim_time), transform=self.ax.transAxes,
                                   fontsize='11')
        # type checking
        for quad_temp in quads:
            if isinstance(quad_temp, Qfm.QuadModel):
                self.quads.append(quad_temp)
            else:
                raise Cf.QuadrotorFlyError("Not a QuadrotorModel type")
        self.draw_grid()

        index = 1
        for quad_temp in self.quads:
            self.traject.append([[], [], []])
            label = self.ax.text([], [], [], 'qua' + str(index), fontsize='11')
            target_point, = self.ax.plot([], [], [], marker='o', color='green', markersize=11, antialiased=False)
            origin_point, = self.ax.plot([], [], [], marker='o', color='black', markersize=6, antialiased=False)
            pos = self.ax.text2D(0.02, 0.87 - 0.09 * index, 'pos_%d/m:' % index + str([0, 0, 0]),
                                 transform=self.ax.transAxes,
                                 fontsize='11')
            attu = self.ax.text2D(0.02, 0.84 - 0.09 * index, 'pos_%d/m:' % index + str([0, 0, 0]),
                                  transform=self.ax.transAxes,
                                  fontsize='11')

            print(target_point, '////target_point')
            index += 1
            if quad_temp.uavPara.structureType == Qfm.StructureType.quad_plus:
                bar_x, = self.ax.plot([], [], [], color='red', linewidth=4, antialiased=False)
                hub, = self.ax.plot([], [], [], marker='o', color='blue', markersize=8, antialiased=False)
                bar_y, = self.ax.plot([], [], [], color='black', linewidth=4, antialiased=False)
                head_x, = self.ax.plot([], [], [], marker='o', color='green', markersize=6, antialiased=False)
                self.quadGui.append({'pos': pos, 'attu': attu, 'hub': hub, 'barX': bar_x, 'barY': bar_y, 'label': label,
                                     'target_point': target_point, 'head_x': head_x, 'origin_point': origin_point})
            elif quad_temp.uavPara.structureType == Qfm.StructureType.quad_x:
                front_bar1, = self.ax.plot([], [], [], color='red', linewidth=4, antialiased=False)
                front_bar2, = self.ax.plot([], [], [], color='red', linewidth=4, antialiased=False)
                hub, = self.ax.plot([], [], [], marker='o', color='blue', markersize=8, antialiased=False)
                back_bar1, = self.ax.plot([], [], [], color='black', linewidth=4, antialiased=False)
                back_bar2, = self.ax.plot([], [], [], color='black', linewidth=4, antialiased=False)
                head_x, = self.ax.plot([], [], [], marker='o', color='green', markersize=6, antialiased=False)
                self.quadGui.append(
                    {'pos': pos, 'attu': attu, 'hub': hub, 'bar_frontLeft': front_bar1, 'bar_frontRight': front_bar2,
                     'bar_rearLeft': back_bar1, 'bar_rearRight': back_bar2, 'label': label,
                     'target_point': target_point, 'head_x': head_x, 'origin_point': origin_point})

    def draw_surface(self):
        x = np.arange(-5, 5, 1)
        y = np.arange(-5, 5, 1)
        x, y = np.meshgrid(x, y)
        z = (0 * x * y + self.target[2])
        self.ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, antialiased=True)

    def draw_grid(self):
        x = np.arange(-4, 5, 1)
        y = np.arange(-4, 5, 1)
        for ik in range(len(x)):
            posx = [-4, 4]
            posy = [y[ik], y[ik]]
            posz = [0, 0]
            figure = self.ax.plot(posx, posy, posz, 'y--')
            posx = [x[ik], x[ik]]
            posy = [-4, 4]
            posz = [0, 0]
            figure = self.ax.plot(posx, posy, posz, 'y--')

    def render(self):
        counts = len(self.quads)
        self.time.set_text('time/s:%.3f' % self.sim_time)
        for ii in range(counts):
            quad = self.quads[ii]
            quad_gui = self.quadGui[ii]
            uav_l = quad.uavPara.uavL
            position = quad.position
            attitude = quad.attitude
            # chang pos_inf
            quad_gui['pos'].set_text('pos_%d/m:' % ii + str('[%.6f, %.6f, %.6f]'
                                                            % (position[0], position[1], position[2])))
            quad_gui['attu'].set_text(('attu_%d/dgree:' % ii + str('[%.6f, %.6f, %.6f]'
                                                                   % (
                                                                       attitude[0] / np.pi * 180,
                                                                       attitude[1] / np.pi * 180,
                                                                       attitude[2] / np.pi * 180))))
            # draw target_point
            quad_gui['target_point'].set_data(self.target[0], self.target[1])
            print(self.target[2])
            quad_gui['target_point'].set_3d_properties(self.target[2])

            # draw origin point
            quad_gui['origin_point'].set_data(0, 0)
            quad_gui['origin_point'].set_3d_properties(0)
            # add traject data
            for ij in range(3):
                self.traject[ii][ij].append(position[ij])
            # move label
            quad_gui['label'].set_position((position[0] + uav_l, position[1]))
            quad_gui['label'].set_3d_properties(position[2] + uav_l, zdir='x')

            # move uav
            if quad.uavPara.structureType == Qfm.StructureType.quad_plus:
                attitude = quad.attitude
                rot_matrix = Cf.get_rotation_matrix(attitude)
                points = np.array([[-uav_l, 0, 0], [uav_l, 0, 0], [0, -uav_l, 0], [0, uav_l, 0], [0, 0, 0]]).T
                points_rotation = np.dot(rot_matrix, points)
                points_rotation[0, :] += position[0]
                points_rotation[1, :] += position[1]
                points_rotation[2, :] += position[2]
                print(points_rotation[0, 0:2], points_rotation[1, 0:2], points_rotation[2, 0:2])
                quad_gui['barX'].set_data(points_rotation[0, 0:2], points_rotation[1, 0:2])
                quad_gui['barX'].set_3d_properties(points_rotation[2, 0:2])
                quad_gui['head_x'].set_data(points_rotation[0, 2], points_rotation[1, 2])
                quad_gui['head_x'].set_3d_properties(points_rotation[2, 2])
                quad_gui['barY'].set_data(points_rotation[0, 2:4], points_rotation[1, 2:4])
                quad_gui['barY'].set_3d_properties(points_rotation[2, 2:4])
                quad_gui['hub'].set_data(points_rotation[0, 4], points_rotation[1, 4])
                quad_gui['hub'].set_3d_properties(points_rotation[2, 4])

            elif quad.uavPara.structureType == Qfm.StructureType.quad_x:
                attitude = quad.attitude
                rot_matrix = Cf.get_rotation_matrix(attitude)
                pos_rotor = uav_l * np.sqrt(0.5)
                # this points is the position of rotor in the body frame; the [0, 0, 0] is the center of UAV;
                #  and the sequence is front_left, front_right, back_left, back_right.
                points = np.array([[pos_rotor, pos_rotor, 0], [0, 0, 0], [pos_rotor, -pos_rotor, 0], [0, 0, 0],
                                   [-pos_rotor, pos_rotor, 0], [0, 0, 0], [-pos_rotor, -pos_rotor, 0], [0, 0, 0]]).T
                # trans axi from body-frame to world-frame
                points_rotation = np.dot(rot_matrix, points)
                points_rotation[0, :] += position[0]
                points_rotation[1, :] += position[1]
                points_rotation[2, :] += position[2]
                quad_gui['bar_frontLeft'].set_data(points_rotation[0, 0:2], points_rotation[1, 0:2])
                quad_gui['bar_frontLeft'].set_3d_properties(points_rotation[2, 0:2])
                quad_gui['head_x'].set_data(points_rotation[0, 2], points_rotation[1, 2])
                quad_gui['head_x'].set_3d_properties(points_rotation[2, 2])
                quad_gui['bar_frontRight'].set_data(points_rotation[0, 2:4], points_rotation[1, 2:4])
                quad_gui['bar_frontRight'].set_3d_properties(points_rotation[2, 2:4])
                quad_gui['bar_rearLeft'].set_data(points_rotation[0, 4:6], points_rotation[1, 4:6])
                quad_gui['bar_rearLeft'].set_3d_properties(points_rotation[2, 4:6])
                quad_gui['bar_rearRight'].set_data(points_rotation[0, 6:8], points_rotation[1, 6:8])
                quad_gui['bar_rearRight'].set_3d_properties(points_rotation[2, 6:8])
                quad_gui['hub'].set_data(position[0], position[1])
                quad_gui['hub'].set_3d_properties(position[2])
            figure = self.ax.plot(self.traject[ii][0], self.traject[ii][1], self.traject[ii][2], c=self.color[ii])
        # if self.ax.elev > 0 and position[2] < 0:
        #     print('11')
        #     self.draw_grid()
        # elif self.ax.elev < 0 and position[2] > 0:
        #     print('22', position)
        #     self.draw_grid()
        # figure = self.ax.plot(pos[0], pos[1], pos[2], c='r')


class QuadrotorFlyGui(object):
    """ Gui manage class"""

    def __init__(self, quads: list):
        self.quads = quads
        self.env = QuadrotorFlyGuiEnv()
        self.ax = self.env.ax
        self.quadGui = QuadrotorFlyGuiUav(self.quads, self.ax)
        self.target = self.quadGui.target

    def render(self):
        self.quadGui.render()
        plt.pause(0.000000000000001)


if __name__ == '__main__':

    " used for testing this module"
    D2R = Qfm.D2R
    testFlag = 1
    if testFlag == 1:
        # import matplotlib as mpl
        print("PID  controller test: ")
        uavPara = Qfm.QuadParas(structure_type=Qfm.StructureType.quad_x)
        simPara = Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed,
                                 init_att=np.array([0, 90, 0]), init_pos=np.array([0, 0, 15]))
        quad1 = Qfm.QuadModel(uavPara, simPara)
        record = MemoryStore.DataRecord()
        record.clear()
        # multi uav test
        quad2 = Qfm.QuadModel(uavPara, simPara)

        # gui init
        gui = QuadrotorFlyGui([quad1])

        # simulation begin
        step_cnt = 0
        pos_x = []
        pos_y = []
        pos_z = []
        pos = []
        ref = np.array([0., 0., -15., 0.])
        err_pos_i = np.array([0, 0, 0])
        for i in range(1000):
            if i == 2000:
                ref = np.array([0., 0., 0., 0])
            if i == 4000:
                ref = np.array([0., 2., 2., 0])
            if i == 6000:
                ref = np.array([4., 2., 5., 0.])
            if i == 8000:
                ref = np.array([2., 0., 2., 0.])
            stateTemp = quad1.observe()
            err_pos_i = err_pos_i + (ref[0:3] - stateTemp[0:3]) * 0.01
            action = quad1.controller_pid(stateTemp, ref)
            action[0] = np.clip(action[0], 0, 20)
            print(action, 'action')
            quad1.step(action)
            pos_x.append(stateTemp[0])
            pos_y.append(stateTemp[1])
            pos_z.append(stateTemp[2])
            pos = [pos_x, pos_y, pos_z]
            # multi uav test
            # action2, oil2 = quad2.get_controller_pid(quad2.observe(), ref)
            # quad2.step(action2)

            if i % 10 == 0:
                gui.quadGui.target = ref[0:3]
                gui.quadGui.sim_time = quad1.ts
                gui.render()
            record.buffer_append((stateTemp, action))
            step_cnt = stateTemp + 1

        record.episode_append()
        print('Quadrotor structure type', quad1.uavPara.structureType)
        # quad1.reset_states()
        print('Quadrotor get reward:', quad1.get_reward())
        data = record.get_episode_buffer()
        bs = data[0]
        ba = data[1]
        t = range(0, record.count)
        # mpl.style.use('seaborn')
        fig1 = plt.figure(2)
        plt.clf()
        plt.subplot(3, 1, 1)
        plt.plot(t, bs[t, 6] / D2R, label='roll')
        plt.plot(t, bs[t, 7] / D2R, label='pitch')
        plt.plot(t, bs[t, 8] / D2R, label='yaw')
        plt.ylabel('Attitude $(\circ)$', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        plt.subplot(3, 1, 2)
        plt.plot(t, bs[t, 0], label='x')
        plt.plot(t, bs[t, 1], label='y')
        plt.ylabel('Position (m)', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        plt.subplot(3, 1, 3)
        plt.plot(t, bs[t, 2], label='z')
        plt.ylabel('Altitude (m)', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        plt.show()
