import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import skimage.io as skio
import cv2

class EnvBrainbow(gym.Env):
    ''' Custom Environment that follows gym interface '''
    metadata = {'render.modes': ['human']}

    def __init__(self, data_volumes, coord_interval, img_mean, img_stddev, num_ch, fov, delta, seed):
        super(EnvBrainbow, self).__init__()
        self.seed(seed)

        # define basic parameters
        self.fov = fov
        self.rad = fov // 2
        self.delta = delta
        self.num_ch = num_ch
        self.dir_map = {'0': np.array([0, delta]), '1': np.array([0, -delta]),
                        '2': np.array([delta, 0]), '3': np.array([-delta, 0])}
        self.rotate_map = {'0': cv2.ROTATE_90_COUNTERCLOCKWISE,
                           '1': cv2.ROTATE_90_CLOCKWISE,
                           '2': cv2.ROTATE_180,
                           '3': None}

        # load image volume and valid coord with skio
        self.img_volume_map, self.coord_map, self.coord_len = self.load(data_volumes, coord_interval, img_mean, img_stddev, num_ch)
        self.num_volume = len(self.img_volume_map)
        print('Preprocessing done. Total', self.num_volume,
              'volumes, Total', self.coord_len, 'reference points per volume.')

        ''' -------------------------------------- '''
        ''' parameters defined every env.reset()   '''
        self.start_point = None  # starting point
        self.start_color = None  # starting color
        self.cur_point = None  # current point
        self.cur_vol = -1  # current volume
        self.cur_ind = -1  # current index
        self.cur_dir = None  # 0 East, 1 West, 2 South, 3 North
        self.cur_dir_offset = None  # dir_map[str(self.cur_dir)]
        self.cur_rotate = None  # rotate_map[str(self.cur_dir)]
        self.offset_list = None  # list that holds offsets from starting point
        self.img_volume = None  # current image volume
        self.img_volume_shape = None  # current image volume shape
        ''' -------------------------------------- '''
        self.observation_space = spaces.Box(low=-4.0, high=4.0, shape=(fov*fov*num_ch,))
        self.action_space = spaces.Discrete(2)  # 0 no move, 1 move
        self.viewer = None

    def step(self, action):
        if action == 0:  # no move
            return np.zeros((self.fov * self.fov * self.num_ch)), 1, True, {}
        elif action == 1:  # move
            # if out of boundary, terminate
            self.cur_point = self.cur_point + self.cur_dir_offset
            if not self.is_fov_boundary(self.img_volume_shape, self.cur_point):
                return np.zeros((self.fov * self.fov * self.num_ch)), 1, True, {}

            self.offset_list.append(self.cur_point)
            rew = 1 if np.all(self.start_color == self.img_volume[self.cur_point[0], self.cur_point[1]]) else -1

            # get patch and rotate the image if required
            patch = self.img_volume[self.cur_point[0] - self.rad:self.cur_point[0] + self.rad + 1,
                    self.cur_point[1] - self.rad:self.cur_point[1] + self.rad + 1]  # (fov, fov, 3)
            if self.cur_rotate is not None:
                patch = cv2.rotate(patch, self.cur_rotate)

            return np.moveaxis(patch, -1, 0).flatten(), rew, False, {}

    def reset(self):
        '''
        Select coordinate randomly
        img volume 0   : p_11, p_12, ..., p_1m
        ...
        img volume n-1 : p_n1, p_n2, ..., p_nm
        order is p_11 -> p_21 -> p_31 -> ... -> p_12 -> ... -> p_nm
        m = max(coord_len_list) duplicated for empty slot
        '''
        # select image volume, coordinate
        self.cur_vol = (self.cur_vol + 1) % self.num_volume
        if self.cur_vol % self.num_volume == 0:
            self.cur_ind = (self.cur_ind + 1) % self.coord_len
        self.start_point = self.coord_map[str(self.cur_vol)][self.cur_ind]

        # select random direction
        self.cur_dir = self.np_random.randint(4)
        self.cur_dir_offset = self.dir_map[str(self.cur_dir)]
        self.cur_rotate = self.rotate_map[str(self.cur_dir)]

        # take image volume
        self.img_volume = self.img_volume_map[str(self.cur_vol)]
        self.img_volume = self.img_volume[self.start_point[0]]  # (y, x, 3)
        self.img_volume_shape = self.img_volume.shape
        self.start_point = self.start_point[1:]  # [y, x]
        self.start_color = self.img_volume[self.start_point[0], self.start_point[1]]

        # start skeletonization
        self.cur_point = self.start_point
        self.offset_list = [self.cur_point]
        patch = self.img_volume[self.cur_point[0]-self.rad:self.cur_point[0]+self.rad+1,
                                self.cur_point[1]-self.rad:self.cur_point[1]+self.rad+1]  # (fov, fov, 3)

        # rotate the image if required
        if self.cur_rotate is not None:
            patch = cv2.rotate(patch, self.cur_rotate)
        return np.moveaxis(patch, -1, 0).flatten()

    def render(self, mode='human'):
        self.img_volume_copy = np.copy(self.img_volume)  # (y, x, 3)
        self.img_volume_copy = (self.img_volume_copy * 33 + 128).astype(np.uint8)

        # draw the starting box and current box
        self.draw_box(self.img_volume_copy, self.cur_point[0], self.cur_point[1],
                      self.rad, color=[252, 255, 51])
        self.draw_box(self.img_volume_copy, self.start_point[0], self.start_point[1],
                      self.rad, color=[240, 128, 128])

        # draw offset in offset_set
        for off in self.offset_list:
            self.img_volume_copy[off[0], off[1]] = [255, 255, 255]

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer(maxwidth=10000)
        self.viewer.imshow(self.img_volume_copy)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        # self.np_random is np.random.RandomState()
        # so, use self.np_random if you want to use
        # random function with constant seed value
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def load(self, data_volumes, coord_interval, img_mean, img_stddev, num_ch):
        '''
        function that returns the dictionary form of normalized data volumes
        and colored coordinates with coord_interval
        {'1' : data_volume1 { with shape (z,y,x,c), dtype np.float32, range [0,1] } ,
         '2' : data_volume2, ... }
        {'1' : valid coords of volume1 { list of coordinates with colored and coord_interval} ,
         '2' : data_volume2, ... }
        '''
        img_volume_map = {}
        coord_map = {}
        coord_len_map = []

        for vol in data_volumes.split(','):
            volname, path = vol.split(':')
            img_volume = skio.imread(path, plugin='tifffile')
            # Check the validity of img_volume
            if img_volume.ndim != 4 or img_volume.dtype != np.uint8 or img_volume.shape[3] != num_ch:
                print('loading image volume ' + volname + ' is not valid.')
                print('shape : ' + str(img_volume.shape) + ' data type : ' + str(img_volume.dtype))
                continue
            z, y, x, _ = img_volume.shape
            assert z == y == x
            # Coordinate processing
            coord = np.argwhere(np.any(img_volume != 0, axis=3))  # take nonzero coordinate
            coord = coord[np.all(coord % coord_interval == 0, axis=1)]  # take only multiples of coord_interval
            coord = coord[np.all(coord >= self.rad, axis=1) & np.all(coord < z-self.rad, axis=1)]  # remove boundary
            self.np_random.shuffle(coord)
            coord_len_map.append(len(coord))
            # Normalization
            img_volume = img_volume.astype(np.float32)
            img_volume = (img_volume - img_mean) / img_stddev
            # Store
            img_volume_map[volname] = img_volume
            coord_map[volname] = coord

        max_len = np.max(coord_len_map)
        for i in range(len(coord_map)):
            coord_map[str(i)] = np.resize(coord_map[str(i)], (max_len, 3))

        return img_volume_map, coord_map, max_len

    def is_fov_boundary(self, img_volume_shape, point):
        ''' Check if the point is in the boundary within fov '''
        bool_y = self.rad - 1 < point[0] < img_volume_shape[0] - self.rad - 1
        bool_x = self.rad - 1 < point[1] < img_volume_shape[1] - self.rad - 1
        return (bool_y and bool_x)

    def draw_box(self, mask, y, x, delta, color):
        '''
        draw box on mask at position (y, x) with size (2*delta, 2*delta)
        with color.

        :param color: list of color range from 0 to 255 ex. [255, 0, 0]
        '''
        if mask.dtype == np.float32:
            color = np.array(color, dtype=np.float32)
            color = color / 255
            mask[y + delta, x - delta:x + delta + 1] = color
            mask[y - delta, x - delta:x + delta + 1] = color
            mask[y - delta:y + delta + 1, x + delta] = color
            mask[y - delta:y + delta + 1, x - delta] = color
        elif mask.dtype == np.uint8:
            color = np.array(color, dtype=np.uint8)
            mask[y + delta, x - delta:x + delta + 1] = color
            mask[y - delta, x - delta:x + delta + 1] = color
            mask[y - delta:y + delta + 1, x + delta] = color
            mask[y - delta:y + delta + 1, x - delta] = color
        else:
            pass