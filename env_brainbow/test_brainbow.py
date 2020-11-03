import argparse
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float


def main():
    num_ch = 3
    fov, delta = 33, 8
    rad = fov // 2
    img_mean, img_stddev = 128., 33.
    img_volume_map = {}

    for vol in args.file.split(','):
        volname, path = vol.split(':')
        img_volume = skio.imread(path, plugin='tifffile')
        # Check the validity of img_volume
        if img_volume.ndim != 4 or img_volume.dtype != np.uint8 or img_volume.shape[3] != num_ch:
            print('loading image volume ' + volname + ' is not valid.')
            print('shape : ' + str(img_volume.shape) + ' data type : ' + str(img_volume.dtype))
            continue
        z, y, x, _ = img_volume.shape
        print(z, y, x, _)

        p = [48, 250, 250]
        p_z, p_y, p_x = p

        plt.imshow(img_volume[p_z, 475:])
        plt.show()

        img_volume_copy = np.copy(img_volume)
        # Normalization
        img_volume = img_volume.astype(np.float32)
        img_volume = (img_volume - img_mean) / img_stddev
        img_plane = img_volume[p_z]
        img_plane_copy = img_volume_copy[p_z]
        patch = img_plane[p_y-rad:p_y+rad+1, p_x-rad:p_x+rad+1]
        patch_copy = img_plane_copy[p_y-rad:p_y+rad+1, p_x-rad:p_x+rad+1]
        patch_slic = slic(patch, n_segments=20, compactness=10, sigma=1, start_label=1)
        patch_copy[rad, rad] = 255

        fig = plt.figure(figsize=(6, 6))
        fig.add_subplot(1, 3, 1)
        plt.imshow(patch_copy)
        fig.add_subplot(1, 3, 2)
        plt.imshow(patch_slic, cmap='nipy_spectral')
        fig.add_subplot(1, 3, 3)
        plt.imshow(mark_boundaries(patch_copy, patch_slic, mode='subpixel'))
        plt.show()
        plt.close()


        # Store
        img_volume_map[volname] = img_volume

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test-brainbow')
    parser.add_argument("--file", type=str, default='0:data/brainbow/training_sample.tif')
    args = parser.parse_args()

    main()