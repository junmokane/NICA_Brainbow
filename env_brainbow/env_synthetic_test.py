from env_brainbow.env_synthetic import EnvBrainbow
import matplotlib.pyplot as plt
import numpy as np
import time


fov, delta = 33, 8
rad = fov // 2
env = EnvBrainbow('0:data/training_sample/training_sample_1.tif'
                  ',1:data/training_sample/training_sample_2.tif'
                  ,
                  coord_interval=4, img_mean=128, img_stddev=33,
                  num_ch=3, fov=fov, delta=delta, seed=0)

obs = env.reset()

for _ in range(10000):
    env.render()
    time.sleep(0.5)
    #fig = plt.figure(figsize=(10, 10))
    #plt.imshow(np.zeros((33, 33, 3)))
    #plt.show()
    #plt.close()
    #plt.imshow((obs * 33 + 128).astype(np.uint8), vmin=0, vmax=255)
    #plt.show()
    #plt.close()
    #time.sleep(0.1)
    '''
    if np.all(obs[rad - delta, rad] == start_color):
        action = 1
    else:
        action = 0
    '''
    obs, rew, done, info = env.step(1)
    print(rew)
    if done:
        obs = env.reset()
        # start_color = obs[rad, rad]

env.close()