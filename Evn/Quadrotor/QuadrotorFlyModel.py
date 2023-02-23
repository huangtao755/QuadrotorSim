#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import enum
from enum import Enum

import numpy as np

from Comman import MemoryStore
from Evn.Quadrotor import SensorCompass, SensorGps, SensorBase, SensorImu

# definition of key constant
D2R = np.pi / 180
state_dim = 12
action_dim = 4
state_bound = np.array([20, 20, 20, 15, 15, 15, 180 * D2R, 80 * D2R, 180 * D2R, 180 * D2R, 180 * D2R, 180 * D2R])
action_bound = np.array([1, 1, 1, 1])


def rk4(func, x0, action, h):
    """Runge Kutta 4 order update function
    :param func: system dynamic
    :param x0: system state
    :param action: control input
    :param h: time of sample
    :return: state of next time
    """
    k1 = func(x0, action)
    k2 = func(x0 + h * k1 / 2, action)
    k3 = func(x0 + h * k2 / 2, action)
    k4 = func(x0 + h * k3, action)
    # print('rk4 debug: ', k1, k2, k3, k4)
    x1 = x0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x1


class StructureType(Enum):
    quad_x = enum.auto()
    quad_plus = enum.auto()


class QuadParas(object):
    """Define the parameters of quadrotor model

    """

    def __init__(self, g=9.81, rotor_num=4, tim_sample=0.01, structure_type=StructureType.quad_plus,
                 uav_l=0.450, uav_m=1.50, uav_ixx=1.75e-2, uav_iyy=1.75e-2, uav_izz=3.18e-2,
                 rotor_ct=1.11e-5, rotor_cm=1.49e-7, rotor_cr=646, rotor_wb=166, rotor_i=9.90e-5, rotor_t=1.36e-2):
        """init the quadrotor parameters
        These parameters are able to be estimation in web(https://flyeval.com/) if you do not have a real UAV.
        common parameters:
            -g          : N/kg,      acceleration gravity
            -rotor-num  : int,       number of rotors, e.g. 4, 6, 8
            -tim_sample : s,         sample time of system
            -structure_type:         quad_x, quad_plus
        uav:
            -uav_l      : m,        distance from center of mass to center of rotor
            -uav_m      : kg,       the mass of quadrotor
            -uav_ixx    : kg.m^2    central principal moments of inertia of UAV in x（惯性矩）
            -uav_iyy    : kg.m^2    central principal moments of inertia of UAV in y
            -uav_izz    : kg.m^2    central principal moments of inertia of UAV in z
        rotor (assume that four rotors are the same):
            -rotor_ct   : N/(rad/s)^2,      lump parameter thrust coefficient, which translate rate of rotor to thrust
            -rotor_cm   : N.m/(rad/s)^2,    lump parameter torque coefficient, like ct, usd in yaw
            -rotor_cr   : rad/s,            scale para which translate oil to rate of motor
            -rotor_wb   : rad/s,            bias para which translate oil to rate of motor
            -rotor_i    : kg.m^2,           inertia of moment of rotor(including motor and propeller)
            -rotor_t    : s,                time para of dynamic response of motor
        """
        self.g = g
        self.numOfRotors = rotor_num
        self.ts = tim_sample
        self.structureType = structure_type
        self.uavL = uav_l
        self.uavM = uav_m
        self.uavInertia = np.array([uav_ixx, uav_iyy, uav_izz])
        self.rotorCt = rotor_ct
        self.rotorCm = rotor_cm
        self.rotorCr = rotor_cr
        self.rotorWb = rotor_wb
        self.rotorInertia = rotor_i
        self.rotorTimScale = 1 / rotor_t


class SimInitType(Enum):
    rand = enum.auto()
    fixed = enum.auto()


class ActuatorMode(Enum):
    simple = enum.auto()
    dynamic = enum.auto()
    disturbance = enum.auto()
    dynamic_voltage = enum.auto()
    disturbance_voltage = enum.auto()


class QuadSimOpt(object):
    """contain the parameters for guiding the simulation process
    """

    def __init__(self, init_mode=SimInitType.rand, init_att=np.array([15, 15, 15]), init_pos=np.array([1, 1, 1]),
                 max_position=20, max_velocity=20, max_attitude=180, max_angular=200,
                 sysnoise_bound_pos=0, sysnoise_bound_att=0,
                 actuator_mode=ActuatorMode.simple, enable_sensor_sys=False):
        """ init the parameters for simulation process, focus on conditions during an episode
        :param init_mode:
        :param init_att:
        :param init_pos:
        :param sysnoise_bound_pos:
        :param sysnoise_bound_att:
        :param actuator_mode:
        :param enable_sensor_sys: whether the sensor system is enable, including noise and bias of sensor
        """
        self.initMode = init_mode
        self.initAtt = init_att
        self.initPos = init_pos
        self.actuatorMode = actuator_mode
        self.sysNoisePos = sysnoise_bound_pos
        self.sysNoiseAtt = sysnoise_bound_att
        self.maxPosition = max_position
        self.maxVelocity = max_velocity
        self.maxAttitude = max_attitude * D2R
        self.maxAngular = max_angular * D2R
        self.enableSensorSys = enable_sensor_sys


class QuadActuator(object):
    """Dynamic of actuator including motor and propeller
    """

    def __init__(self, quad_para: QuadParas, mode: ActuatorMode):
        """Parameters is maintain together
        :param quad_para:   parameters of quadrotor,maintain together
        :param mode:        'simple', without dynamic of motor; 'dynamic' with dynamic;
        """
        self.para = quad_para
        self.motorPara_scale = self.para.rotorTimScale * self.para.rotorCr
        self.motorPara_bias = self.para.rotorTimScale * self.para.rotorWb
        self.mode = mode

        # states of actuator
        self.outThrust = np.zeros([self.para.numOfRotors])
        self.outTorque = np.zeros([self.para.numOfRotors])
        # rate of rotor
        self.rotorRate = np.zeros([self.para.numOfRotors])

    def dynamic_actuator(self, rotor_rate, action):
        """dynamic of motor and propeller
        input: rotorRate, u
        output: rotorRateDot,
        """

        rate_dot = self.motorPara_scale * action + self.motorPara_bias - self.para.rotorTimScale * rotor_rate
        return rate_dot

    def reset(self):
        """reset all state"""

        self.outThrust = np.zeros([self.para.numOfRotors])
        self.outTorque = np.zeros([self.para.numOfRotors])
        # rate of rotor
        self.rotorRate = np.zeros([self.para.numOfRotors])

    def step(self, action: 'int > 0'):
        """calculate the next state based on current state and u
        :param action:
        :return:
        """
        action = np.clip(action, 0, 1)
        # if u > 1:
        #     u = 1

        if self.mode == ActuatorMode.simple:
            # without dynamic of motor
            self.rotorRate = self.para.rotorCr * action + self.para.rotorWb
        elif self.mode == ActuatorMode.dynamic:
            # with dynamic of motor
            self.rotorRate = rk4(self.dynamic_actuator, self.rotorRate, action, self.para.ts)
        else:
            self.rotorRate = 0

        self.outThrust = self.para.rotorCt * np.square(self.rotorRate)
        self.outTorque = self.para.rotorCm * np.square(self.rotorRate)
        return self.outThrust, self.outTorque


class QuadModel(object):
    """module interface, main class including basic dynamic of quad
    """

    def __init__(self, uav_para: QuadParas, sim_para: QuadSimOpt):
        """init a quadrotor
        :param uav_para:    parameters of quadrotor,maintain together
        :param sim_para:    'simple', without dynamic of motor; 'dynamic' with dynamic;
        """
        self.uavPara = uav_para
        self.simPara = sim_para
        self.actuator = QuadActuator(self.uavPara, sim_para.actuatorMode)

        # states of quadrotor
        #   -position, m
        self.position = np.array([0, 0, 0])
        #   -velocity, m/s
        self.velocity = np.array([0, 0, 0])
        #   -attitude, rad
        self.attitude = np.array([0, 0, 0])
        #   -angular, rad/s
        self.angular = np.array([0, 0, 0])
        # accelerate, m/(s^2)
        self.acc = np.zeros(3)

        # time control, s
        self.__ts = 0

        # initial the sensors
        if self.simPara.enableSensorSys:
            self.sensorList = list()
            self.imu0 = SensorImu.SensorImu()
            self.gps0 = SensorGps.SensorGps()
            self.mag0 = SensorCompass.SensorCompass()
            self.sensorList.append(self.imu0)
            self.sensorList.append(self.gps0)
            self.sensorList.append(self.mag0)

        # initial the states
        self.reset_states()


    @property
    def ts(self):
        """return the tick of system"""
        return self.__ts

    def generate_init_att(self):
        """used to generate a init attitude according to simPara"""
        angle = self.simPara.initAtt * D2R
        if self.simPara.initMode == SimInitType.rand:
            phi = (1 * np.random.random() - 0.5) * angle[0]
            theta = (1 * np.random.random() - 0.5) * angle[1]
            psi = (1 * np.random.random() - 0.5) * angle[2]
        else:
            phi = angle[0]
            theta = angle[1]
            psi = angle[2]
        return np.array([phi, theta, psi])

    def generate_init_pos(self):
        """used to generate a init position according to simPara"""
        pos = self.simPara.initPos
        if self.simPara.initMode == SimInitType.rand:
            x = (1 * np.random.random() - 0.5) * pos[0]
            y = (1 * np.random.random() - 0.5) * pos[1]
            z = (1 * np.random.random() - 0.5) * pos[2]
        else:
            x = pos[0]
            y = pos[1]
            z = pos[2]
        return np.array([x, y, z])

    def reset_states(self, att='none', pos='none'):
        self.__ts = 0
        self.actuator.reset()
        if isinstance(att, str):
            self.attitude = self.generate_init_att()
        else:
            self.attitude = att

        if isinstance(pos, str):
            self.position = self.generate_init_pos()
        else:
            self.position = pos

        self.velocity = np.array([0, 0, 0])
        self.angular = np.array([0, 0, 0])

        # sensor system reset
        if self.simPara.enableSensorSys:
            for sensor in self.sensorList:
                sensor.reset(self.state)

    def dynamic_basic(self, state, action):
        """ calculate /dot(state) = f(state) + u(state)
        This function will be executed many times during simulation, so high performance is necessary.
        :param state:
            0       1       2       3       4       5
            p_x     p_y     p_z     v_x     v_y     v_z
            6       7       8       9       10      11
            roll    pitch   yaw     v_roll  v_pitch v_yaw
        :param action: u1(sum of thrust), u2(torque for roll), u3(pitch), u4(yaw)
        :return: derivatives of state inclfrom bokeh.plotting import figure
        """
        # variable used repeatedly
        att_cos = np.cos(state[6:9])
        att_sin = np.sin(state[6:9])
        noise_pos = self.simPara.sysNoisePos * np.random.random(3)
        noise_att = self.simPara.sysNoiseAtt * np.random.random(3)

        dot_state = np.zeros([12])
        # dynamic of position cycle
        dot_state[0:3] = state[3:6]
        # ########### we need not to calculate the whole rotation matrix because just care last column
        dot_state[3:6] = action[0] / self.uavPara.uavM * np.array([
            att_cos[2] * att_sin[1] * att_cos[0] + att_sin[2] * att_sin[0],
            att_sin[2] * att_sin[1] * att_cos[0] - att_cos[2] * att_sin[0],
            att_cos[0] * att_cos[1]
        ]) - np.array([0, 0, self.uavPara.g]) + noise_pos

        # dynamic of attitude cycle
        dot_state[6:9] = state[9:12]
        # Coriolis force on UAV from motor, this is affected by the direction of rotation.
        #   Pay attention, it needs to be modify when the model of uav varies.
        #   The signals of this equation should be same with toque for yaw
        rotor_rate_sum = (self.actuator.rotorRate[3] + self.actuator.rotorRate[2]
                          - self.actuator.rotorRate[1] - self.actuator.rotorRate[0])

        para = self.uavPara
        dot_state[9:12] = np.array([
            state[10] * state[11] * (para.uavInertia[1] - para.uavInertia[2]) / para.uavInertia[0]
            - para.rotorInertia / para.uavInertia[0] * state[10] * rotor_rate_sum
            + para.uavL * action[1] / para.uavInertia[0],

            state[9] * state[11] * (para.uavInertia[2] - para.uavInertia[0]) / para.uavInertia[1]
            + para.rotorInertia / para.uavInertia[1] * state[9] * rotor_rate_sum
            + para.uavL * action[2] / para.uavInertia[1],

            state[9] * state[10] * (para.uavInertia[0] - para.uavInertia[1]) / para.uavInertia[2]
            + action[3] / para.uavInertia[2]
        ]) + noise_att

        ''' Just used for test
        temp1 = state[10] * state[11] * (para.uavInertia[1] - para.uavInertia[2]) / para.uavInertia[0]
        temp2 = - para.rotorInertia / para.uavInertia[0] * state[10] * rotor_rate_sum
        temp3 = + para.uavL * action[1] / para.uavInertia[0]
        print('dyanmic Test', temp1, temp2, temp3, action)
       '''

        return dot_state

    def observe(self):
        """out put the system state, with sensor system or without sensor system"""
        if self.simPara.enableSensorSys:
            sensor_data = dict()
            for index, sensor in enumerate(self.sensorList):
                if isinstance(sensor, SensorBase.SensorBase):
                    # name = str(index) + '-' + sensor.get_name()
                    name = sensor.get_name()
                    sensor_data.update({name: sensor.observe()})
            print(sensor_data)
            return sensor_data
        else:
            return np.hstack([self.position, self.velocity, self.attitude, self.angular])

    @property
    def  state(self):
        return np.hstack([self.position, self.velocity, self.attitude, self.angular])

    def is_finished(self):
        if (np.max(np.abs(self.position)) < self.simPara.maxPosition) \
                and (np.max(np.abs(self.velocity) < self.simPara.maxVelocity)) \
                and (np.max(np.abs(self.attitude) < self.simPara.maxAttitude)) \
                and (np.max(np.abs(self.angular) < self.simPara.maxAngular)):
            return False
        else:
            return True

    def get_reward(self):
        reward = np.sum(np.square(self.position)) / 8 + np.sum(np.square(self.velocity)) / 20 \
                 + np.sum(np.square(self.attitude)) / 3 + np.sum(np.square(self.angular)) / 10
        return reward

    def rotor_distribute_dynamic(self, action):
        x_shape = 0
        pluse_shape = np.array([[+1, +1, +1, +1],
                      [-1, +1, +1, -1],
                      [+1, +1, -1, -1],
                      [+1, +1, -1, -1]])

        self.actuator.rotorRate[3]
        pass
        """ calculate torque according to the distribution of rotors
        :param thrusts:
        :param torque:
        :return:
        """
        ''' The structure of quadrotor, left is '+' and the right is 'x'
        The 'x' 'y' in middle defines the positive direction X Y axi in body-frame, which is a right hand frame.
        The numbers inside the rotors indicate the index of the motors;
        The signals show the direction of rotation, positive is ccw while the negative is cw.
        ---------------------------------------------------------------------------------------------------
                        ******                                                                            
                      **  3   **                                          ****                 ****     
                     **   -    **                                       **    **             **    **   
                      **      **                                      **   3    **         **    1   ** 
                       **    **                                       **   -    **         **    +   **  
                          **                                            **    **             **    **   
            ****          **          ****                                ****   **   **   **  ****     
         **      **     **  **      **    **            x(+)                       ***  ***            
        **   2    **  **      **  **   1    **       y(+)  y(-)                   **      **              
        **   +    **  **      **  **   +    **          x(-)                      **      **              
         **     **      ******      **    **                                      * **  ** *           
            ****          **          ****                                ****  **          ** ****    
                          **                                            **    **             **    **  
                        **  **                                        **   2    **         **    4   **
                      **      **                                      **   +    **         **    -   **
                     **   4    **                                       **    **             **    **  
                      **  -   **                                          ****                 ****    
                        ******     
        ---------------------------------------------------------------------------------------------------                                                                          
        '''
        # forces = np.zeros(4)
        # if self.uavPara.structureType == StructureType.quad_plus:
        #     forces[0] = np.sum(thrusts)
        #     forces[1] = thrusts[1] - thrusts[0]
        #     forces[2] = thrusts[3] - thrusts[2]
        #     forces[3] = torque[3] + torque[2] - torque[1] - torque[0]
        # elif self.uavPara.structureType == StructureType.quad_x:
        #     forces[0] = np.sum(thrusts)
        #     forces[1] = -thrusts[0] + thrusts[1] + thrusts[2] - thrusts[3]
        #     forces[2] = -thrusts[0] + thrusts[1] - thrusts[2] + thrusts[3]
        #     forces[3] = -torque[0] - torque[1] + torque[2] + torque[3]
        # else:
        #     forces = np.zeros(4)


    # def step(self, action: 'int > 0'):
    #
    #     self.__ts += self.uavPara.ts
    #     # 1.1 Actuator model, calculate the thrust and torque
    #     thrusts, toques = self.actuator.step(action)
    #
    #     # 1.2 Calculate the force distribute according to 'x' type or '+' type, assum '+' type
    #     forces = self.rotor_distribute_dynamic(thrusts, toques)
    #
    #     # 1.3 Basic model, calculate the basic model, the u need to be given directly in test-mode for Matlab
    #     state_temp = np.hstack([self.position, self.velocity, self.attitude, self.angular])
    #     state_next = rk4(self.dynamic_basic, state_temp, forces, self.uavPara.ts)
    #     [self.position, self.velocity, self.attitude, self.angular] = np.split(state_next, 4)
    #     # calculate the accelerate
    #     state_dot = self.dynamic_basic(state_temp, forces)
    #     self.acc = state_dot[3:6]
    #
    #     # 2. Calculate Sensor sensor model
    #     if self.simPara.enableSensorSys:
    #         for index, sensor in enumerate(self.sensorList):
    #             if isinstance(sensor, SensorBase.SensorBase):
    #                 sensor.update(np.hstack([state_next, self.acc]), self.__ts)
    #     ob = self.observe()
    #
    #     # 3. Check whether finish (failed or completed)
    #     finish_flag = self.is_finished()
    #
    #     # 4. Calculate a reference reward
    #     reward = self.get_reward()
    #
    #     return ob, reward, finish_flag

    def step(self, action: 'int > 0'):
        print(self.__ts)

        self.__ts += self.uavPara.ts
        # 1.1 Actuator model, calculate the thrust and torque
        # thrusts, toques = self.actuator.step(action)

        # 1.2 Calculate the force distribute according to 'x' type or '+' type, assum '+' type
        # forces = self.rotor_distribute_dynamic(thrusts, toques)

        # 1.3 Basic model, calculate the basic model, the u need to be given directly in test-mode for Matlab
        state_temp = np.hstack([self.position, self.velocity, self.attitude, self.angular])
        state_next = rk4(self.dynamic_basic, state_temp, action, self.uavPara.ts)
        [self.position, self.velocity, self.attitude, self.angular] = np.split(state_next, 4)
        # calculate the accelerate
        state_dot = self.dynamic_basic(state_temp, action)
        self.acc = state_dot[3:6]

        # 2. Calculate Sensor sensor model
        if self.simPara.enableSensorSys:
            for index, sensor in enumerate(self.sensorList):
                if isinstance(sensor, SensorBase.SensorBase):
                    sensor.update(np.hstack([state_next, self.acc]), self.__ts)
        ob = self.observe()

        # 3. Check whether finish (failed or completed)
        finish_flag = self.is_finished()

        # 4. Calculate a reference reward
        reward = self.get_reward()

        return ob, reward, finish_flag

    def controller_pid(self, state, ref_state=np.array([0, 0, 0, 0])):
        """

        :param state:
        :param ref_state:
        :return:
        """
        action = np.zeros(4)

        # position-velocity cycle, velocity cycle is regard as kd
        ki_pos = np.array([0.0, 0.0, 0.0])
        kp_pos = np.array([0.43, 0.43, 0.6])
        kp_vel = np.array([1.01, 1.01, 1.4])

        # calculate a_pos
        err_pos = ref_state[0:3] - state[0:3]
        err_vel = - state[3:6]
        a_pos = kp_pos * err_pos + kp_vel * err_vel
        a_pos[2] = a_pos[2] + self.uavPara.g
        u1 = self.uavPara.uavM * np.sqrt(sum(np.square(a_pos)))

        # attitude-angular cycle, angular cycle is regard as kd
        kp_angle = np.array([15.5, 15.5, 15])
        kp_angular = np.array([7.7, 7.7, 7])

        # calculate a_angle
        phi = state[6]
        theta = state[7]
        phy = state[8]

        phy_ref = ref_state[3]
        phi_ref = np.arcsin(self.uavPara.uavM * (a_pos[0] * np.sin(phy_ref) - a_pos[1] * np.cos(phy_ref)) / u1)
        theta_ref = np.arcsin(
            self.uavPara.uavM * (a_pos[0] * np.cos(phy_ref) + a_pos[1] * np.sin(phy_ref)) / (u1 * np.cos(phi_ref)))
        angle = np.array([phi, theta, phy])
        # print(angle, 'angle')
        angle_ref = np.array((phi_ref, theta_ref, phy_ref))
        angle_err = angle_ref - angle
        angle_vel_err = -state[9:12]
        a_angle = kp_angle * angle_err + kp_angular * angle_vel_err
        u2 = a_angle[0] * self.uavPara.uavInertia[0]
        u3 = a_angle[1] * self.uavPara.uavInertia[1]
        u4 = a_angle[2] * self.uavPara.uavInertia[2]
        action = np.array([u1, u2, u3, u4])

        if self.uavPara.structureType == StructureType.quad_plus:
            pass
        elif self.uavPara.structureType == StructureType.quad_x:
            pass

        return action

    def get_controller_pid(self, state, err_pos_i=np.array([0, 0, 0]),
                           ref_state=np.array([0, 0, 1, 0])):
        """ pid controller
        :param state: system state, 12
        :param ref_state: reference value for x, y, z, yaw
        :return: control value for four motors
        """

        # position-velocity cycle, velocity cycle is regard as kd
        ki_pos = np.array([0.0, 0.0, 0.0])
        kp_pos = np.array([1, 1, 1])
        kp_vel = np.array([0.15, 0.15, 0.5])

        # decoupling about x-y
        phy = state[8]
        # de_phy = np.array([[np.sin(phy), -np.cos(phy)], [np.cos(phy), np.sin(phy)]])
        # de_phy = np.array([[np.cos(phy), np.sin(phy)], [np.sin(phy), -np.cos(phy)]])
        de_phy = np.array([[np.cos(phy), -np.sin(phy)], [np.sin(phy), np.cos(phy)]])
        err_pos = ref_state[0:3] - np.array([state[0], state[1], state[2]])
        ref_vel = err_pos * kp_pos  # + err_pos_i * ki_pos

        err_vel = ref_vel - np.array([state[3], state[4], state[5]])

        # calculate ref without decoupling about phy
        # ref_angle = kp_vel * err_vel
        # calculate ref with decoupling about phy
        ref_angle = np.zeros(3)
        ref_angle[0:2] = np.matmul(de_phy, kp_vel[0] * err_vel[0:2])

        # attitude-angular cycle, angular cycle is regard as kd
        kp_angle = np.array([50, 50, 40])
        kp_angular = np.array([0.1, 0.1, 0.1])
        # ref_angle = np.zeros(3)
        err_angle = np.array([-ref_angle[1], ref_angle[0], ref_state[3]]) - np.array([state[6], state[7], state[8]])
        ref_rate = err_angle * kp_angle
        err_rate = ref_rate - [state[9], state[10], state[11]]
        con_rate = err_rate * kp_angular

        # the control value in z direction needs to be modify considering gravity
        err_altitude = (ref_state[2] - state[2]) * 0.5
        con_altitude = (err_altitude - state[5]) * 0.1
        # print(con_altitude)
        oil_altitude = 0.634195 + con_altitude
        # oil_altitude = 0.6 + con_altitude
        if oil_altitude > 0.75:
            oil_altitude = 0.75

        action_motor = np.zeros(4)
        if self.uavPara.structureType == StructureType.quad_plus:
            action_motor[0] = oil_altitude - con_rate[0] - con_rate[2]
            action_motor[1] = oil_altitude + con_rate[0] - con_rate[2]
            action_motor[2] = oil_altitude - con_rate[1] + con_rate[2]
            action_motor[3] = oil_altitude + con_rate[1] + con_rate[2]
        elif self.uavPara.structureType == StructureType.quad_x:
            action_motor[0] = oil_altitude - con_rate[2] - con_rate[1] - con_rate[0]
            action_motor[1] = oil_altitude - con_rate[2] + con_rate[1] + con_rate[0]
            action_motor[2] = oil_altitude + con_rate[2] - con_rate[1] + con_rate[0]
            action_motor[3] = oil_altitude + con_rate[2] + con_rate[1] - con_rate[0]
        else:
            action_motor = np.zeros(4)

        action_pid = action_motor
        return action_pid, oil_altitude


if __name__ == '__main__':
    " used for testing this module"
    testFlag = 3

    if testFlag == 1:
        # test for actuator
        qp = QuadParas()
        ac0 = QuadActuator(qp, ActuatorMode.simple)
        print("QuadActuator Test")
        print("dynamic result0:", ac0.rotorRate)
        result1 = ac0.dynamic_actuator(ac0.rotorRate, np.array([0.2, 0.4, 0.6, 0.8]))

        print("dynamic result1:", result1)
        result2 = ac0.dynamic_actuator(np.array([400, 800, 1200, 1600]), np.array([0.2, 0.4, 0.6, 0.8]))
        print("dynamic result2:", result2)
        ac0.reset()
        ac0.step(np.array([0.2, 0.4, 0.6, 0.8]))
        print("dynamic result3:", ac0.rotorRate, ac0.outTorque, ac0.outThrust)
        print("QuadActuator Test Completed! ---------------------------------------------------------------")
    elif testFlag == 2:
        print("Basic model test: ")
        uavPara = QuadParas()
        simPara = QuadSimOpt(init_mode=SimInitType.fixed, actuator_mode=ActuatorMode.dynamic,
                             init_att=np.array([0.2, 0.2, 0.2]), init_pos=np.array([0, 0, 0]))
        quad1 = QuadModel(uavPara, simPara)
        u = np.array([100., 20., 20., 20.])
        stateTemp = np.array([1., 2., 3., 0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6])
        result1 = quad1.dynamic_basic(stateTemp, np.array([100., 20., 20., 20.]))
        print("result1 ", result1)
        [quad1.pos, quad1.velocity, quad1.attitude, quad1.angular] = np.split(stateTemp, 4)
        result2 = quad1.step(u)
        print("result2 ", result2, quad1.pos, quad1.velocity, quad1.attitude, quad1.angular)
    elif testFlag == 3:
        import matplotlib.pyplot as plt

        print("PID  controller test: ")
        uavPara = QuadParas(structure_type=StructureType.quad_x)
        simPara = QuadSimOpt(init_mode=SimInitType.fixed, enable_sensor_sys=False,
                             init_att=np.array([10., -10., 30]), init_pos=np.array([15, -15, 10]))
        quad1 = QuadModel(uavPara, simPara)
        record = MemoryStore.DataRecord()
        record.clear()
        step_cnt = 0
        for i in range(3000):
            ref = np.array([0., 3., 0., 0.5])
            stateTemp = quad1.observe()
            action2, oil = quad1.get_controller_pid(stateTemp, ref)
            print('action: ', action2)
            action2 = np.clip(action2, 0.1, 0.9)
            quad1.step(action2)
            record.buffer_append((stateTemp, action2))
            step_cnt = step_cnt + 1
        record.episode_append()

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
