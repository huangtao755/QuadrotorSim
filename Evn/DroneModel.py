import enum
from enum import Enum

import numpy as np

from matplotlib import pyplot as plt

from Comman import MemoryStore
from Evn.Quadrotor import SensorCompass, SensorGps, SensorBase, SensorImu

# definition of key constant
D2R = np.pi / 180
state_dim = 12
action_dim = 4

# 再讨论 #
state_bound = np.array([20, 20, 20, 5, 5, 5, 180 * D2R, 180 * D2R, 180 * D2R, 90 * D2R, 90 * D2R, 90 * D2R])
action_bound = np.array([1, 1, 1, 1])


def rk4(func, x0, u, h):
    """Runge Kutta 4 order update function
    :param func: system dynamic
    :param x0: system state
    :param oil: control input
    :param h: time of sample
    :return: state of next time
    """
    k1 = func(x0, u)
    k2 = func(x0 + h * k1 / 2, u)
    k3 = func(x0 + h * k2 / 2, u)
    k4 = func(x0 + h * k3, u)
    # print('rk4 debug: ', k1, k2, k3, k4)
    x1 = x0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x1


class StructureType(Enum):
    quad_x = enum.auto()
    quad_plus = enum.auto()


class SimInitType(Enum):
    rand = enum.auto()
    fixed = enum.auto()


class ActuatorMode(Enum):
    simple = enum.auto()
    dynamic = enum.auto()
    disturbance = enum.auto()
    dynamic_voltage = enum.auto()
    disturbance_voltage = enum.auto()


class QuadParas(object):
    """Define the parameters of quadrotor model

    """
    def __init__(self,
                 g=9.81,
                 rotor_num=4,
                 tim_sample=0.01,
                 structure_type=StructureType.quad_plus,
                 uav_l=0.225,
                 uav_m=1.50,
                 uav_ixx=1.318e-2,
                 uav_iyy=1.318e-2,
                 uav_izz=2.365e-2,
                 rotor_ct=1.253e-5,
                 rotor_cm=1.852e-7,
                 rotor_cr=578.95,
                 rotor_wb=147.64,
                 rotor_i=5.51e-5,
                 rotor_t=0.0095,
                 k_p=6.579e-2,
                 k_a=9.012e-3):
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
        self.bodyKp = k_p
        self.bodyKa = k_a


class QuadSimOpt(object):
    """contain the parameters for guiding the simulation process
    """

    def __init__(self,
                 init_mode=SimInitType.rand,
                 init_att=np.array([15, 15, 15]),
                 init_pos=np.array([1, 1, 1]),
                 max_position=20,
                 max_velocity=20,
                 max_attitude=180,
                 max_angular=200,
                 sysnoise_bound_pos=0,
                 sysnoise_bound_att=0,
                 actuator_mode=ActuatorMode.simple,
                 enable_sensor_sys=False):
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

        # rate of rotor
        self.rotorRate = np.zeros([self.para.numOfRotors])

    def dynamic_actuator(self, rotor_rate, oil):
        """dynamic of motor and propeller
        input: rotorRate, u
        output: rotorRateDot,
        """
        rate_dot = self.motorPara_scale * oil + self.motorPara_bias - self.para.rotorTimScale * rotor_rate
        return rate_dot
    
    def reset(self):
        """reset all state"""
        # rate of rotor
        self.rotorRate = np.zeros([self.para.numOfRotors])
        
    def step(self, oil: 'int > 0'):
        """calculate the next state based on current state and u
        :param oil:
        :return:
        """
        oil = np.clip(oil, 0, 1)
        # if u > 1:
        #     u = 1

        if self.mode == ActuatorMode.simple:
            # without dynamic of motor
            self.rotorRate = self.para.rotorCr * oil + self.para.rotorWb
        elif self.mode == ActuatorMode.dynamic:
            # with dynamic of motor
            self.rotorRate = rk4(self.dynamic_actuator, self.rotorRate, oil, self.para.ts/10)
        else:
            self.rotorRate = 0

        return self.rotorRate

    # 单纯pid不好用


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

        # 世界系到载体系坐标变化
        self.rotor_mat = np.zeros((3, 3))

        self.gain_mat_plus = np.linalg.inv(np.narray([[self.uavPara.rotorCt, self.uavPara.rotorCt,
                                                     self.uavPara.rotorCt, self.uavPara.rotorCt],
                                                     [0, self.uavPara.uavL*self.uavPara.rotorCt,
                                                     0, -self.uavPara.uavL*self.uavPara.rotorCt],
                                                     [-self.uavPara.uavL*self.uavPara.rotorCt, 0,
                                                     self.uavPara.uavL*self.uavPara.rotorCt, 0],
                                                     [self.uavPara.rotorCm, -self.uavPara.rotorCm,
                                                     self.uavPara.rotorCm, -self.uavPara.rotorCm]]))

        self.gain_mat_x = np.linalg.inv(np.narray([[self.uavPara.rotorCt,
                                                  self.uavPara.rotorCt,
                                                  self.uavPara.rotorCt,
                                                  self.uavPara.rotorCt],
                                                  [-self.uavPara.uavL/np.sqrt(2)*self.uavPara.rotorCt,
                                                  -self.uavPara.uavL/np.sqrt(2)*self.uavPara.rotorCt,
                                                  self.uavPara.uavL/np.sqrt(2)*self.uavPara.rotorCt,
                                                  self.uavPara.uavL/np.sqrt(2)*self.uavPara.rotorCt],
                                                  [-self.uavPara.uavL/np.sqrt(2)*self.uavPara.rotorCt,
                                                  self.uavPara.uavL/np.sqrt(2)*self.uavPara.rotorCt,
                                                  self.uavPara.uavL/np.sqrt(2)*self.uavPara.rotorCt,
                                                  -self.uavPara.uavL/np.sqrt(2)*self.uavPara.rotorCt],
                                                  [self.uavPara.rotorCm,
                                                  -self.uavPara.rotorCm,
                                                  self.uavPara.rotorCm,
                                                  -self.uavPara.rotorCm]]))

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

    def rotor_distribute_dynamic(self, action):
        action = action.reshape((4, 1))
        if self.uavPara.structureType == StructureType.quad_plus:
            motor_rate_square = (self.gain_mat_plus @ action).reshape(4)    # mat(4*4) @ mat(4*1) -> mat(4*1) 变为 1*4 的向量
        elif self.uavPara.structureType == StructureType.quad_x:
            motor_rate_square = (self.gain_mat_x @ action).reshape(4)
        else:
            motor_rate_square = np.zeros(4)

        return motor_rate_square

    def dynamic_basic(self, state, action:  np.array):
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
        para = self.uavPara

        # get motor rotor rate
        motor_rate = np.sqrt(self.rotor_distribute_dynamic(action))
        rotor_rate_sum = np.array([1, -1, 1, -1]) @ motor_rate.T

        # variable used repeatedly
        att_cos = np.cos(state[6:9])
        att_sin = np.sin(state[6:9])
        noise_pos = self.simPara.sysNoisePos * np.random.random(3)
        noise_att = self.simPara.sysNoiseAtt * np.random.random(3)

        dot_state = np.zeros([12])
        # dynamic of position cycle
        dot_state[0:3] = state[3:6]
        # ########### we need not to calculate the whole rotation matrix because just care last column
        # 输入力， 重力， 阻力， 扰动
        dot_state[3:6] = action[0] / para.uavM * np.array([
            att_cos[2] * att_sin[1] * att_cos[0] + att_sin[2] * att_sin[0],
            att_sin[2] * att_sin[1] * att_cos[0] - att_cos[2] * att_sin[0],
            att_cos[0] * att_cos[1]
        ]) \
            - np.array([0, 0, para.g]) \
            - np.array([para.bodyKp, para.bodyKp, para.bodyKp]) @ state[3:6].T / para.uavM \
            + noise_pos / para.uavM

        # dynamic of attitude cycle
        # Coriolis force on UAV from motor, this is affected by the direction of rotation.
        #   Pay attention, it needs to be modify when the model of uav varies.
        #   The signals of this equation should be same with toque for yaw

        # ############# 计算机体系下姿态速度
        self.rotor_mat = np.array([1, att_sin[1]*att_sin[0]/att_cos[1], att_sin[1]*att_cos[0]/att_cos[1]],
                                  [0, att_cos[0], -att_sin[0]],
                                  [0, att_sin[0]/att_cos[1], att_cos[0]/att_cos[1]])
        state_body = (np.linalg.inv(self.rotor_mat) @ state[9:12].T).reshape(3)             # 获取机体角速度

        # ############# 计算集体角加速度

        dot_state[9:12] = np.array([
            state_body[1] * state[2] * (para.uavInertia[1] - para.uavInertia[2]) / para.uavInertia[0]
            - para.rotorInertia / para.uavInertia[0] * state[2] * rotor_rate_sum
            + action[1] / para.uavInertia[0] - para.bodyKa/para.uavInertia[0]*state_body[0],

            state_body[0] * state_body[2] * (para.uavInertia[2] - para.uavInertia[0]) / para.uavInertia[1]
            + para.rotorInertia / para.uavInertia[1] * state_body[0] * rotor_rate_sum
            + action[2] / para.uavInertia[1] - para.bodyKa/para.uavInertia[1]*state_body[1],

            state_body[0] * state_body[1] * (para.uavInertia[0] - para.uavInertia[1]) / para.uavInertia[2]
            + action[3] / para.uavInertia[2] - para.bodyKa/para.uavInertia[2]*state_body[2]
        ]) + noise_att

        dot_state[6:9] = (state_body @ self.rotor_mat.T).reshape(3)

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
    def state(self):
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

    def step(self, action):
        print(self.__ts)

        self.__ts = self.uavPara.ts

        state_temp = np.hstack([self.position, self.velocity, self.attitude, self.angular])
        state_body_next = rk4(self.dynamic_basic, state_temp, action, self.uavPara.ts)
        [self.position, self.velocity, self.attitude, self.angular] = np.split(state_body_next, 4)








