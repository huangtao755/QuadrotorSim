#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""'used for manage and generate cam data with set ground-image.
it is a virtual camera

By xiaobo
Contact linxiaobo110@gmail.com
Created on  十一月 21 20:29 2019
"""

# Copyright (C)
#
# This file is part of quadrotorfly
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.


import enum
from enum import Enum

import cv2
import numpy as np
from numba import jit

"""
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: CamDown
**  Module Date: 2019/11/21
**  Module Auth: xiaobo
**  Version    : V0.1
**  Description: a virtual camera looking down
**-------------------------------------------------------------------------------------------------------
**  Reversion  :
**  Modified By:
**  Date       :
**  Content    :
**  Notes      :
********************************************************************************************************/
"""


class CamDownPara(Enum):
    Render_Mode_Mem = enum.auto()
    Render_Mode_Cpu = enum.auto()
    Render_Mode_Gpu = enum.auto()


class CamDown(object):
    def __init__(self, img_horizon=400, img_vertical=400, img_depth=3,
                 sensor_horizon=4., sensor_vertical=4., cam_focal=2.36,
                 ground_img_path='./Data/groundImgWood.jpg',
                 small_ground_img_path='./Data/groundImgSmall.jpg',
                 small_land_img_path='./Data/landingMark.jpg',
                 render_mode=CamDownPara.Render_Mode_Mem):
        """
        ****************** horizon ****************
        *
        v
        e
        r
        t
        i
        c
        a
        l
        ********************************************
        :param img_horizon: the num of vertical pixes of the image which is generated by sensor
        :param img_vertical: the num of horizon pixes of the image which is generated by sensor
        :param img_depth: the num of chanel the image, i.e. rgb is 3
        :param sensor_horizon:   mm,     the height of the active area of the sensor
        :param sensor_vertical:  mm,     the width of the active area of the sensor
        :param cam_focal:       mm,     the focal of the lens of the camera
        """
        self.imgHorizon = img_horizon
        self.imgVertical = img_vertical
        self.imgDepth = img_depth
        self.skx = sensor_horizon * 1. / img_horizon  # skx is sensor_k_x
        self.sky = sensor_vertical * 1. / img_vertical  # skx is sensor_k_y
        self.sx0 = sensor_horizon * 0.5
        self.sy0 = sensor_vertical * 0.5
        self.camFocal = cam_focal
        self.axCamImgArr = np.zeros([self.imgVertical * self.imgHorizon, 3])
        self.pixCamImg = np.zeros([self.imgVertical, self.imgHorizon, self.imgDepth], dtype=np.uint8)
        self.groundImgPath = ground_img_path
        self.groundImg = None

        # small image
        self.smallGroundImgPath = small_ground_img_path
        self.smallLandingImgPath = small_land_img_path
        self.smallGroundImg = None
        self.smallLandingImg = None
        self.smallImgHorizon = 7854
        self.smallImgVertical = 3490
        self.smallLandingImgHos = 315
        self.smallLandingImgVet = 315
        self.renderMode = render_mode

        # init the array
        for ii in range(self.imgVertical):
            for j in range(self.imgHorizon):
                self.axCamImgArr[ii * self.imgHorizon + j, :] = [ii, j, 1]

    def load_ground_img(self):
        if self.renderMode == CamDownPara.Render_Mode_Mem:
            if self.groundImg is not None:
                del self.groundImg
            else:
                self.groundImg = cv2.imread(self.groundImgPath)
        elif self.renderMode == CamDownPara.Render_Mode_Cpu:
            if self.smallGroundImg is not None:
                del self.smallGroundImg
            else:
                self.smallGroundImg = cv2.imread(self.smallGroundImgPath)
                # if read success, get size of image
                if self.smallGroundImg is not None:
                    self.smallImgVertical, self.smallImgHorizon = self.smallGroundImg.shape
        elif self.renderMode == CamDownPara.Render_Mode_Gpu:
            if self.smallGroundImg is not None:
                del self.smallGroundImg
            else:
                self.smallGroundImg = cv2.imread(self.smallGroundImgPath)
                self.smallLandingImg = cv2.imread(self.smallLandingImgPath)
                # if read success, get size of image
                if self.smallGroundImg is not None:
                    self.smallImgVertical, self.smallImgHorizon, _ = self.smallGroundImg.shape
                if self.smallLandingImg is not None:
                    self.smallLandingImgVet, self.smallLandingImgHos, _ = self.smallLandingImg.shape

    def get_img_by_state(self, pos, att):
        m_img2sensor = np.array([[self.skx, 0, self.sx0],
                                 [0, -self.sky, -self.sy0],
                                 [0, 0, 1]])
        m_sensor2cam = np.array([[pos[2] / self.camFocal, 0, self.imgVertical / 2],
                                 [0, pos[2] / self.camFocal, -self.imgHorizon / 2],
                                 [0, 0, 1]])
        m_cam2world = Cf.get_rotation_matrix(att)
        m_cam2world[0:2, 2] = pos[0:2]
        m_cam2world[2, :] = np.array([0, 0, 1])
        m_trans = m_cam2world.dot(m_sensor2cam.dot(m_img2sensor))
        if self.renderMode == CamDownPara.Render_Mode_Mem:
            ax_real = m_trans.dot(self.axCamImgArr.transpose()).transpose()
            for ii in range(self.imgVertical):
                for j in range(self.imgHorizon):
                    self.pixCamImg[ii, j, :] = self.groundImg[int(ax_real[ii * self.imgHorizon + j, 0]),
                                               int(ax_real[ii * self.imgHorizon + j, 1] + 10000), :]
        elif self.renderMode == CamDownPara.Render_Mode_Cpu:
            ax_real = m_trans.dot(self.axCamImgArr.transpose()).transpose()
            for ii in range(self.imgVertical):
                for j in range(self.imgHorizon):
                    ax_vertical = int(ax_real[ii * self.imgHorizon + j, 0]) % self.smallImgVertical
                    ax_horizon = int(ax_real[ii * self.imgHorizon + j, 1] + 10000) % self.smallImgHorizon
                    self.pixCamImg[ii, j, :] = self.smallGroundImg[ax_vertical, ax_horizon, :]

        elif self.renderMode == CamDownPara.Render_Mode_Gpu:
            ax_real = m_trans.dot(self.axCamImgArr.transpose()).transpose()
            accelerate_img_mapping_gpu(ax_real, self.smallGroundImg, self.smallLandingImg, self.pixCamImg,
                                       self.imgVertical, self.imgHorizon, self.smallImgHorizon, self.smallImgVertical,
                                       self.smallLandingImgHos, self.smallLandingImgVet)
        return self.pixCamImg


@jit(nopython=True)
def accelerate_img_mapping_gpu(m_ax_real, m_img_small, m_img_landing, m_img_result,
                               img_vertical, img_horizon, small_horizon, small_vertical, land_horizon, land_vertical):
    """
    # 用来加速映射计算的函数
    :param m_ax_real: 映射后的座标
    :param m_img_small: 贴图，小图片
    :param m_img_result: 目标图像
    :param m_img_landing
    :param img_vertical: 目标图像高度
    :param img_horizon: 目标图像宽度
    :param small_horizon: 贴图宽度
    :param small_vertical: 贴图高度
    :param land_horizon:
    :param land_vertical
    :return:
    """
    land_hos_max = 5000 + land_horizon
    land_vet_min = 5000 - land_vertical
    for ii in range(img_vertical):
        for j in range(img_horizon):
            ax_vertical = int(m_ax_real[ii * img_horizon + j, 0])
            ax_horizon = int(m_ax_real[ii * img_horizon + j, 1] + 10000)
            if (ax_horizon > 5000) and (ax_horizon < land_hos_max) and (ax_vertical > land_vet_min) \
                    and (ax_vertical < 5000):
                ax_horizon = ax_horizon - 5000
                ax_vertical = ax_vertical - land_vet_min
                m_img_result[ii, j, :] = m_img_landing[ax_vertical, ax_horizon, :]
            else:
                ax_vertical = ax_vertical % small_vertical
                ax_horizon = ax_horizon % small_horizon
                m_img_result[ii, j, :] = m_img_small[ax_vertical, ax_horizon, :]
    return m_img_result


if __name__ == '__main__':
    " used for testing this module"
    testFlag = 3

    if testFlag == 1:
        cam1 = CamDown()
        print('init completed!')
        cam1.load_ground_img()
        print('Load img completed!')
        pos_0 = np.array([4251.97508977, -5843.00458236, 227.35483937])
        att_0 = np.array([-0.13647458, -0.1990263, 0.27836947])
        img1 = cam1.get_img_by_state(pos_0, att_0)
        import matplotlib.pyplot as plt

        plt.imshow(cam1.pixCamImg / 255)
        cv2.imwrite('Data/test.jpg', img1)

    elif testFlag == 2:
        import matplotlib.pyplot as plt
        # import matplotlib as mpl
        from QuadrotorFlyModel import QuadModel, QuadSimOpt, QuadParas, StructureType, SimInitType
        from Comman import MemoryStore, Utils as Cf

        D2R = np.pi / 180

        print("PID  controller test: ")
        uavPara = QuadParas(structure_type=StructureType.quad_x)
        simPara = QuadSimOpt(init_mode=SimInitType.fixed, enable_sensor_sys=False,
                             init_att=np.array([10., -10., 30]), init_pos=np.array([5, -5, 0]))
        quad1 = QuadModel(uavPara, simPara)
        record = MemoryStore.DataRecord()
        record.clear()
        step_cnt = 0

        # init the camera
        cam1 = CamDown()
        cam1.load_ground_img()
        print('Load img completed!')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out1 = cv2.VideoWriter('Data/img/test.avi', fourcc, 1 / quad1.uavPara.ts, (cam1.imgVertical, cam1.imgHorizon))
        for i in range(1000):
            ref = np.array([0., 0., 1., 0.])
            stateTemp = quad1.observe()
            # get image
            pos_0 = quad1.position * 1000
            att_0 = quad1.attitude
            img1 = cam1.get_img_by_state(pos_0, att_0)
            # file_name = 'Data/img/test_' + str(i) + '.jpg'
            # cv2.imwrite(file_name, img1)
            out1.write(img1)

            action2, oil = quad1.get_controller_pid(stateTemp, ref)
            print('action: ', action2)
            action2 = np.clip(action2, 0.1, 0.9)
            quad1.step(action2)
            record.buffer_append((stateTemp, action2))
            step_cnt = step_cnt + 1
        record.episode_append()
        out1.release()

        print('Quadrotor structure type', quad1.uavPara.structureType)
        # quad1.reset_states()
        print('Quadrotor get reward:', quad1.get_reward())
        data = record.get_episode_buffer()
        bs = data[0]
        ba = data[1]
        t = range(0, record.count)
        # mpl.style.use('seaborn')
        fig1 = plt.figure(1)
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
    # performance test
    elif testFlag == 3:
        import matplotlib.pyplot as plt
        # import matplotlib as mpl
        from QuadrotorFlyModel import QuadModel, QuadSimOpt, QuadParas, StructureType, SimInitType
        import time

        D2R = np.pi / 180
        video_write_flag = True

        print("PID  controller test: ")
        uavPara = QuadParas(structure_type=StructureType.quad_x)
        simPara = QuadSimOpt(init_mode=SimInitType.fixed, enable_sensor_sys=False,
                             init_att=np.array([0., 0., 0]), init_pos=np.array([0, -0, 0]))
        quad1 = QuadModel(uavPara, simPara)
        record = MemoryStore.DataRecord()
        record.clear()
        step_cnt = 0

        # init the camera
        cam1 = CamDown(render_mode=CamDownPara.Render_Mode_Gpu)
        cam1.load_ground_img()
        print('Load img completed!')
        if video_write_flag:
            v_format = cv2.VideoWriter_fourcc(*'MJPG')
            out1 = cv2.VideoWriter('Data/img/test.avi', v_format, 1 / quad1.uavPara.ts,
                                   (cam1.imgVertical, cam1.imgHorizon))
        for i in range(1000):
            if i == 0:
                time_start = time.time()
            ref = np.array([0., 0., 3., 0.])
            stateTemp = quad1.observe()
            # get image
            pos_0 = quad1.position * 1000
            att_0 = quad1.attitude
            img1 = cam1.get_img_by_state(pos_0, att_0)
            # file_name = 'Data/img/test_' + str(i) + '.jpg'
            # cv2.imwrite(file_name, img1)
            if video_write_flag:
                out1.write(img1)

            action2, oil = quad1.get_controller_pid(stateTemp, ref)
            print('action: ', action2)
            action2 = np.clip(action2, 0.1, 0.9)
            quad1.step(action2)
            record.buffer_append((stateTemp, action2))
            step_cnt = step_cnt + 1
        time_end = time.time()
        print('time cost:', str(time_end - time_start))
        record.episode_append()
        if video_write_flag:
            out1.release()

        print('Quadrotor structure type', quad1.uavPara.structureType)
        # quad1.reset_states()
        print('Quadrotor get reward:', quad1.get_reward())
        data = record.get_episode_buffer()
        bs = data[0]
        ba = data[1]
        t = range(0, record.count)
        # mpl.style.use('seaborn')
        fig1 = plt.figure(1)
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
