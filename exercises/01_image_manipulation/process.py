#!/usr/bin/env python3

import os
import sys

import numpy as np

from PIL import Image

SHRINK_FACTOR = 5

# TODO: Use fact that gaussian kernels separable for more efficient processing
# Stolen straight from the lecture notes ;)
BLUR_KERNEL_SMALL_X = np.array([[1, 2, 1]]) / 16
BLUR_KERNEL_SMALL_Y = np.array([[1], [2], [1]])
BLUR_KERNEL_SMALL = np.array([[1,2,1], [2,4,2], [1,2,1]])/16

BLUR_KERNEL_LARGE_X = np.array([[0.834,3.739,6.164,3.739,0.834]]) / 234
BLUR_KERNEL_LARGE_Y = np.array([[0.834],[3.739],[6.164],[3.3739],[0.834]])
BLUR_KERNEL_LARGE = np.array(
        [
            [1,3,5,3,1],
            [3,14,23,14,3],
            [5,23,38,23,5],
            [3,14,23,14,3],
            [1,3,5,3,1],
        ]
) / 234

SOBEL_KERNEL_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
SOBEL_KERNEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])


def main():
    if len(sys.argv) != 4:
        usage()

    operation = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    try:
        with Image.open(input_file) as img:
            pixels = image_to_pixels(img)

            if operation == 'resize':
                out_pixels = shrink(pixels, SHRINK_FACTOR, get_nearest_neighbour)
            elif operation == 'blur':
                out_pixels = convolute_2D(pixels, BLUR_KERNEL_LARGE)
            elif operation == 'edge':
                # Need to ensure we can square without overflowing
                out_x = convolute_2D(pixels, SOBEL_KERNEL_X).astype(np.uint16)
                out_y = convolute_2D(pixels, SOBEL_KERNEL_Y).astype(np.uint16)
                # And after the square root we truncate back to uint8.
                out_pixels = np.sqrt(np.square(out_x) + np.square(out_y)).astype(np.uint8)
            elif operation == 'edge_x':
                out_pixels = convolute_2D(pixels, SOBEL_KERNEL_X)
            elif operation == 'edge_y':
                out_pixels = convolute_2D(pixels, SOBEL_KERNEL_Y)
            else:
                usage()

            out_img = pixels_to_image(out_pixels)
            out_img.save(output_file)

    except OSError as e:
        print("Error: {}".format(e))

# Convert (height * width) Pillow image into a (height, width, 3) numpy array.
def image_to_pixels(img):
    return np.array(img)

# Convert numpy (height, width, 3) array into (height * width) Pillow image
def pixels_to_image(pixels):
    return Image.fromarray(pixels)

def convolute_2D(pixels, kernel):
    input_height = pixels.shape[0]
    input_width = pixels.shape[1]

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    output = np.zeros(pixels.shape, dtype=np.uint8)

    # With an nxn kernel, the middle field is for the pixel itself, so we need
    # floor(n/2) padding on each side.
    pad_width = int(kernel_width / 2)
    padded = edge_pad_2D(pixels, pad_width)

    # TODO should be able to use np.fromfunction or similar here
    for y in range(0, input_height):
        for x in range(0, input_width):
            # Corresponding pixel in padded image is offset by `pad_width` in
            # either direction. But as we want a window *centered* on the given
            # pixel, we subtract `pad_width` again, so it cancels out.
            window = padded[y : y + kernel_height, x : x + kernel_width]

            # Treat R, G, B separately
            for ch in range(3):
                # We want component-wise multiplication of kernel and (sub)-matrix of the
                # image, so use np.multiply (which is what the * operator does), then sum
                # the entries up.
                output[y, x, ch] = (kernel * window[:, :, ch]).sum()

    return output

# Takes a (height, width, 3) array representing an RGB image, and
# two-dimensionally edge-pads it along the height and width axes.
#
# Returns a (height + pad_width, width + pad_width, 3) array.
#
# Bootleg version of np.pad(ary, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), dtype=int)
# Probably less performant, but hey.
def edge_pad_2D(ary, pad_width):
    height = ary.shape[0]
    width = ary.shape[1]

    # Initiate zero-filled array with pad_width additional rows / columns on
    # every side.
    out = np.zeros((height + 2 * pad_width, width + 2 * pad_width, 3), dtype=int)
    # And place the input flush in the middle
    out[pad_width:-pad_width, pad_width:-pad_width] = ary

    # Edge padding rows first, then columns (or also the other way around)
    # allows not worrying about the non-orthogonal fields, as they will come
    # naturally.

    # Edge pad top and bottom rows
    for i in range(pad_width):
        out[i] = out[pad_width]
        # Mind that last row is -1
        out[-i - 1] = out[-pad_width - 1]

    # Edge pad left and right columns
    for i in range(pad_width):
        out[:, i] = out[:, pad_width]
        out[:, -i - 1] = out[:, -pad_width - 1]

    return out

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

def shrink(pixels, factor, sampling_function):
    height, width, _ = pixels.shape
    # New shape is 1/factor for width and height, and still RGB
    new_shape = (int(height / factor), int(width / factor), 3)

    shrunk_pixels = np.fromfunction(
            sampling_function,
            new_shape,
            dtype=int,
            src=pixels,
            scale_factor=factor,
    )

    return shrunk_pixels


def usage():
    print ("Usage: process.py <resize|blur|edge|edge_x|edge_y> <input_file> <output_file>")
    sys.exit(1)

if __name__ == '__main__':
    main()
