from Evn.CartPole import CartPoleEnv

import time

env = CartPoleEnv()
env.reset()
print(env.sys_matr())
# env.step(env.action_space.sample())
# for _ in range(1000):
#     env.render()
#     action = env.action_space.sample()
#     state, done = env.step(0.1)
#     print(action, type(action))
#     time.sleep(0.01)
#     if done:
#         env.reset()
