from env_brainbow.env_brainbow import EnvBrainbow
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import time


fov, delta, num_ch = 33, 8, 3
rad = fov // 2
env = EnvBrainbow('0:data/brainbow/training_sample_1.tif',
                  coord_interval=1, img_mean=128, img_stddev=33,
                  num_ch=3, fov=fov, delta=delta, seed=0)

obs = env.reset()
#start_color = (np.moveaxis(np.reshape(obs, (num_ch, fov, fov)), 0, -1)[rad, rad]) * 33 + 128
start_color = env.start_color

for _ in range(10000):
    env.render()
    #time.sleep(0.5)
    obs_copy = np.copy(obs)
    obs_copy = np.reshape(obs_copy, (num_ch, fov, fov))
    obs_copy = np.moveaxis(obs_copy, 0, -1)
    obs_copy = (obs_copy * 33 + 128).astype(np.uint8)
    #print(obs_copy[rad, rad])

    #fig = plt.figure(figsize=(10, 10))
    #plt.imshow(obs_copy, vmin=0, vmax=255)
    #plt.show()
    #plt.close()
    '''
    if np.all(obs[rad - delta, rad] == start_color):
        action = 1
    else:
        action = 0
    '''
    cur_color = (np.moveaxis(np.reshape(obs, (num_ch, fov, fov)), 0, -1)[rad-delta, rad]) * 33 + 128
    cosine_val = (cur_color @ start_color.T) / (norm(cur_color) * norm(start_color))
    print('start color', start_color, 'current color', cur_color)
    print('cosine value', cosine_val)
    if np.any(cur_color >= 30) and cosine_val >= 0.9:
        action = 1
    else:
        action = 0

    obs, rew, done, info = env.step(action)
    print('reward', rew)
    if done:
        obs = env.reset()
        #start_color = (np.moveaxis(np.reshape(obs, (num_ch, fov, fov)), 0, -1)[rad, rad]) * 33 + 128
        start_color = env.start_color

env.close()