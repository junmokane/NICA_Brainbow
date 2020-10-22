from env_brainbow.env_brainbow import EnvBrainbow
import time
import matplotlib.pyplot as plt
import numpy as np

fov, delta = 33, 8
env = EnvBrainbow('0:data/training_sample/training_sample_1.tif'
                  ',1:data/training_sample/training_sample_2.tif'
                  ,
                  coord_interval=4, img_mean=128, img_stddev=33, num_ch=3, fov=fov, delta=delta, seed=0)

obs = env.reset()

for _ in range(10000):
    env.render()
    #fig = plt.figure(figsize=(10, 10))
    #plt.imshow((obs * 33 + 128).astype(np.uint8), vmin=0, vmax=255)
    #plt.show()
    #plt.close()
    #time.sleep(0.1)
    obs, rew, done, info = env.step(1)
    if done:
        obs = env.reset()

env.close