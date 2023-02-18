#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math

import numpy as np
import matplotlib.pyplot as plt


from Comman import MemoryStore


D2R = np.pi / 180
current_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))


class SpringCar(object):
    def __init__(self, name='SpringCar'):
        self.name = name

        self.gravity = 9.8
        self.masscart = 1.0
        

        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_done = None
        self.record = MemoryStore.DataRecord()
        self.record.clear()
        self.record_flag = False

    def step(self, action, record=True):
        # err_msg = f"{action!r} ({type(action)}) invalid"
        # assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        # force = self.force_mag if action == 1 else -self.force_mag
        force = action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta -self.mu_cart * x_dot
        ) / self.total_mass
        thetaacc = ((self.gravity * sintheta - costheta * temp) - self.mu_pole*theta_dot) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if record:
            self.record.buffer_append((self.state, action))
        return np.array(self.state, dtype=np.float32), done

    def reset(
        self,
        *,
        init_state=None,
        return_info: bool = False
    ):
        if init_state is None:
            self.state = np.random.uniform(low=-0.5, high=0.5, size=(4,))
            self.steps_beyond_done = None
        else:
            self.state = init_state

        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def sys_matr(self):
        j = self.masscart * self.length**2 / 3
        a_22 = - self.mu_cart * (j + self.masspole * self.length**2) / \
            (j * self.total_mass + self.masscart * self.masspole * self.length**2)
        a_23 = - self.masspole**2 * self.length**2 * self.gravity / \
            (j * self.total_mass + self.masscart * self.masspole * self.length**2)
        a_42 = self.mu_cart * self.masspole * self.length / \
            (j * self.total_mass + self.masscart * self.masspole * self.length**2)
        a_43 = self.masspole * self.length * self.gravity * (self.masspole + self.masscart) / \
            (j * self.total_mass + self.masscart * self.masspole * self.length**2)
        A = np.zeros((4, 4))
        # A = np.array([[0, 1, 0, 0], [0, a_22, a_23, 0], [0, 0, 0, 1], [0, a_42, a_43, 0]])
        A[0, 1] = 1
        A[1, 1] = a_22
        A[1, 2] = a_23
        A[2, 3] = 1
        A[3, 1] = a_42
        A[3, 2] = a_43

        b_2 = (j + self.masspole * self.length**2) / \
            (j * self.total_mass + self.masscart * self.masspole * self.length**2)
        b_4 = - (self.masspole * self.length) / \
            (j * self.total_mass + self.masscart * self.masspole * self.length**2)
        B = np.zeros((4, 1))
        B[1, 0] = b_2
        B[3, 0] = b_4

        return A, B

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False

    def data_save(self, reward=None):
        """

        :return:
        """
        data = self.record.get_episode_buffer()
        state_data = data[0]
        action_data = data[1]

        self.record.save_data(path=current_path + '//DataSave//CartPole', data_name=str(self.name + '_state'),
                              data=state_data)
        self.record.save_data(path=current_path + '//DataSave//CartPole', data_name=str(self.name + '_action'),
                              data=action_data)

    def fig_show(self, i=1):
        """

        :param i:
        :return:
        """
        if self.record_flag is False:
            self.record.episode_append()
            self.data_save()

        data = self.record.get_episode_buffer()
        bs = data[0]
        ba = data[1]
        t = range(0, self.record.count)
        ts = np.array(t) * self.tau

        self.fig1 = plt.figure(int(1+i**2))
        plt.clf()
        plt.subplot(4, 1, 1)
        plt.plot(ts, bs[t, 0], label='x')
        plt.ylabel('Position (m)', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        plt.subplot(4, 1, 2)
        plt.plot(ts, bs[t, 1], label='x')
        plt.ylabel('Velocity (m/s)', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        plt.subplot(4, 1, 3)
        plt.plot(ts, bs[t, 2], label='z')
        plt.ylabel('Angle $(\circ)$', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        plt.subplot(4, 1, 4)
        plt.plot(ts, bs[t, 3], label='z')
        plt.ylabel('Angle_v $(\circ)$/s', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))

        self.fig2 = plt.figure((2+i**2))
        plt.clf()
        plt.plot(ts, ba)
        plt.ylabel('u', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        # plt.show()
