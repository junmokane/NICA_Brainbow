from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
import numpy as np
import matplotlib.pyplot as plt
from rlkit.core import logger
import skimage.io as skio
import cv2
from collections import deque

filename = str(uuid.uuid4())


def is_fov_boundary(img_plane_shape, rad, point):
    ''' Check if the point is in the boundary within fov '''
    bool_y = rad - 1 < point[0] < img_plane_shape[0] - rad - 1
    bool_x = rad - 1 < point[1] < img_plane_shape[1] - rad - 1
    return (bool_y and bool_x)


def draw_box(mask, point, rad, color):
    '''
    draw box on mask at position (y, x) with size (2*delta, 2*delta)
    with color.

    :param color: list of color range from 0 to 255 ex. [255, 0, 0]
    '''
    if is_fov_boundary(mask.shape, rad, point):
        y, x = point
        if mask.dtype == np.float32:
            color = np.array(color, dtype=np.float32)
            color = color / 255
            mask[y + rad, x - rad:x + rad + 1] = color
            mask[y - rad, x - rad:x + rad + 1] = color
            mask[y - rad:y + rad + 1, x + rad] = color
            mask[y - rad:y + rad + 1, x - rad] = color
        elif mask.dtype == np.uint8:
            color = np.array(color, dtype=np.uint8)
            mask[y + rad, x - rad:x + rad + 1] = color
            mask[y - rad, x - rad:x + rad + 1] = color
            mask[y - rad:y + rad + 1, x + rad] = color
            mask[y - rad:y + rad + 1, x - rad] = color
        else:
            pass
    else:
        pass


def simulate_policy(args):
    fov, delta, num_ch = 33, 8, 3
    rad = fov // 2
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    env = data['evaluation/env']
    print("Policy loaded")

    img_volume = skio.imread('data/training_sample/training_sample_1.tif', plugin='tifffile')
    img_volume_copy = np.copy(img_volume)
    img_volume = img_volume.astype(np.float32)
    img_volume = (img_volume - 128) / 33

    s_point = [0, 190, 250]
    s_z, s_y, s_x = s_point
    img_plane = img_volume[s_z].astype(np.float32)
    img_plane_copy = img_volume_copy[s_z]

    s_point_list = []

    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    policy.reset()  # does noting

    # initialization
    Q = deque([[s_y, s_x]])
    V = [[s_y, s_x]]
    color = [240, 128, 128]

    while len(Q) > 0:
        c_y, c_x = Q.popleft()  # current y, x

        cur_p_t = img_plane[c_y - rad:c_y + rad + 1, c_x - rad:c_x + rad + 1]  # current patch top
        cur_p_l = cv2.rotate(cur_p_t, cv2.ROTATE_90_CLOCKWISE)  # current patch left
        cur_p_r = cv2.rotate(cur_p_t, cv2.ROTATE_90_COUNTERCLOCKWISE)  # current patch right
        cur_p_b = cv2.rotate(cur_p_t, cv2.ROTATE_180)  # current patch bottom

        a_t, _ = policy.get_action(np.moveaxis(cur_p_t, -1, 0).flatten())  # move top
        a_l, _ = policy.get_action(np.moveaxis(cur_p_l, -1, 0).flatten())  # move left
        a_r, _ = policy.get_action(np.moveaxis(cur_p_r, -1, 0).flatten())  # move right
        a_b, _ = policy.get_action(np.moveaxis(cur_p_b, -1, 0).flatten())  # move bottom

        top = [c_y - delta, c_x]
        left = [c_y, c_x - delta]
        right = [c_y, c_x + delta]
        bottom = [c_y + delta, c_x]

        if a_t == 1:
            img_plane_copy[c_y - delta:c_y + 1, c_x] = color
            if top not in V:
                Q.append(top)
                V.append(top)
        if a_l == 1:
            img_plane_copy[c_y, c_x - delta:c_x + 1] = color
            if left not in V:
                Q.append(left)
                V.append(left)
        if a_r == 1:
            img_plane_copy[c_y, c_x:c_x + delta + 1] = color
            if right not in V:
                Q.append(right)
                V.append(right)
        if a_b == 1:
            img_plane_copy[c_y:c_y + delta + 1, c_x] = color
            if bottom not in V:
                Q.append(bottom)
                V.append(bottom)

    img_plane_copy[s_y, s_x] = [252, 255, 51]  # starting point color
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img_plane_copy)
    plt.show()
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--de', type=bool, default=False,
                        help='stop and detect failure case.')
    parser.add_argument('--ep', type=int, default=1000,
                        help='# of episodes to run')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--S', type=float, default=1,
                        help='time sleep when rendering')
    parser.add_argument('--gpu', action='store_false')
    args = parser.parse_args()

    simulate_policy(args)


'''
cur_p_t = img_plane[s_y - rad:s_y + rad + 1, s_x - rad:s_x + rad + 1]  # current patch top
cur_p_l = cv2.rotate(cur_p_t, cv2.ROTATE_90_CLOCKWISE)  # current patch left
cur_p_r = cv2.rotate(cur_p_t, cv2.ROTATE_90_COUNTERCLOCKWISE)  # current patch right
cur_p_b = cv2.rotate(cur_p_t, cv2.ROTATE_180)  # current patch bottom

a_t, _ = policy.get_action(np.moveaxis(cur_p_t, -1, 0).flatten())  # move top
a_l, _ = policy.get_action(np.moveaxis(cur_p_l, -1, 0).flatten())  # move left
a_r, _ = policy.get_action(np.moveaxis(cur_p_r, -1, 0).flatten())  # move right
a_b, _ = policy.get_action(np.moveaxis(cur_p_b, -1, 0).flatten())  # move bottom
print(a_t, a_l, a_r, a_b)

fig = plt.figure(figsize=(10, 10))
fig.add_subplot(2, 2, 1)
plt.imshow((cur_p_t * 33 + 128).astype(np.uint8))
fig.add_subplot(2, 2, 2)
plt.imshow((cur_p_l * 33 + 128).astype(np.uint8))
fig.add_subplot(2, 2, 3)
plt.imshow((cur_p_r * 33 + 128).astype(np.uint8))
fig.add_subplot(2, 2, 4)
plt.imshow((cur_p_b * 33 + 128).astype(np.uint8))
plt.show()
plt.close()
'''