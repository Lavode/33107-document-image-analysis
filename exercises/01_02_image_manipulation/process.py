#!/usr/bin/env python3

import os
import sys

import numpy as np

from PIL import Image

SHRINK_FACTOR = 5

def main():
    if len(sys.argv) != 3:
        usage()

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        with Image.open(input_file) as img:
            out_img = shrink(img, SHRINK_FACTOR, get_nearest_neighbour)
            out_img.save(output_file)

    except OSError as e:
        print("Error: {}".format(e))

# Convert (height * width) Pillow image into a (height, width, 3) numpy array.
def image_to_pixels(img):
    return np.array(img)

# Convert numpy (height, width, 3) array into (height * width) Pillow image
def pixels_to_image(pixels):
    return Image.fromarray(pixels)

# Nearest-neighbour downsampling.
#
# Given a coordinate (x, y) and source image src, return the pixel at (x /
# scale_factor + offset, y / scale_factor + offset).  The offset, if left
# unspecified, is calculated as scale_factor / 2, leading to a pixel 'in the
# middle' of each square to be chosen. If set to 0, the top-left pixel is
# chosen instead.
def get_nearest_neighbour(x, y, channel, src=None, scale_factor=None, offset=None):
    if src is None:
        raise RuntimeError("get_nearest_neighbour: src must not be None")

    if scale_factor is None:
        raise RuntimeError("get_nearest_neighbour: scale_factor must not be None")

    if offset is None:
        # Default to a center pixel
        offset = int(scale_factor / 2)

    new_x = x * scale_factor + offset
    new_y = y * scale_factor + offset

    return src[new_x, new_y, channel]

def shrink(img, factor, sampling_function):
    width, height = img.size
    # New shape is 1/factor for width and height, and still RGB
    new_shape = (int(height / factor), int(width / factor), 3)

    pixels = image_to_pixels(img)
    shrunk_pixels = np.fromfunction(
            sampling_function,
            new_shape,
            dtype=int,
            src=pixels,
            scale_factor=factor,
    )

    return pixels_to_image(shrunk_pixels)


def usage():
    print ("Usage: resize.py <input_file> <output_file>")
    sys.exit(1)

if __name__ == '__main__':
    main()
