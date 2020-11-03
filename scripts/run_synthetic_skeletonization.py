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
    # hyper-parameters
    fov, delta, num_ch = 33, 8, 3
    rad = fov // 2
    data = torch.load(args.file)
    color = [240, 128, 128]

    # load policy & env
    policy = data['evaluation/policy']
    env = data['evaluation/env']
    print("Policy loaded")
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    policy.reset()  # does noting

    # load image
    img_volume = skio.imread('data/training_sample/training_sample_1.tif', plugin='tifffile')
    img_volume_copy = np.copy(img_volume)
    img_volume = img_volume.astype(np.float32)
    img_volume = (img_volume - 128) / 33

    # select specific z slice
    s_z = 0
    img_plane = img_volume[s_z].astype(np.float32)
    img_plane_copy = img_volume_copy[s_z]
    img_plane_shape = img_plane.shape

    # gather random starting point for each color
    # random starting point should be colored and also not FoV boundary
    s_point_list = []  # list of starting point
    unique_colors = np.unique(np.reshape(img_plane, (-1, 3)), axis=0)
    for color in unique_colors:
        if np.all(color == (-128. / 33)):
            continue
        color_index_list_tmp = np.argwhere(np.all(img_plane == color, axis=2))
        #print(color_index_list_tmp)
        color_index_list = []
        for index in color_index_list_tmp:
            if is_fov_boundary(img_plane_shape, rad, index):
                color_index_list.append(index)
        #print(color_index_list)
        #print(type(color_index_list))
        len_color_index_list = len(color_index_list)
        if len_color_index_list > 0:
            random_index = np.random.choice(len_color_index_list, 1)
            random_start_point = color_index_list[random_index[0]]
            #print(random_start_point)
            s_point_list.append(random_start_point)
    # print(s_point_list)

    # start skeletonization
    for i, s_point in enumerate(s_point_list):
        print('Skeletoninzing', i+1, 'th point:', s_point)
        # initialization
        img_plane_tmp = np.copy(img_plane_copy)
        s_y, s_x = s_point
        Q = deque([[s_y, s_x]])
        V = [[s_y, s_x]]

        # start skeletonization for some starting point
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
                if top not in V and is_fov_boundary(img_plane_shape, rad, top):
                    img_plane_tmp[c_y - delta:c_y + 1, c_x] = color
                    Q.append(top)
                    V.append(top)
            if a_l == 1:
                if left not in V and is_fov_boundary(img_plane_shape, rad, left):
                    img_plane_tmp[c_y, c_x - delta:c_x + 1] = color
                    Q.append(left)
                    V.append(left)
            if a_r == 1:
                if right not in V and is_fov_boundary(img_plane_shape, rad, right):
                    img_plane_tmp[c_y, c_x:c_x + delta + 1] = color
                    Q.append(right)
                    V.append(right)
            if a_b == 1:
                if bottom not in V and is_fov_boundary(img_plane_shape, rad, bottom):
                    img_plane_tmp[c_y:c_y + delta + 1, c_x] = color
                    Q.append(bottom)
                    V.append(bottom)

        # plot final result
        img_plane_tmp[s_y-1:s_y+2, s_x-1:s_x+2] = [252, 255, 51]  # color starting point
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(img_plane_tmp)
        plt.show()
        plt.close()






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
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