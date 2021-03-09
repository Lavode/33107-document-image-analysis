#!/usr/bin/env python3

import os
import sys

import numpy as np

import cv2

SHRINK_FACTOR = 5

BLUR_KERNEL_SMALL = np.array([[1,2,1], [2,4,2], [1,2,1]])/16
BLUR_KERNEL_LARGE = np.array(
        [
            [1,3,5,3,1],
            [3,14,23,14,3],
            [5,23,38,23,5],
            [3,14,23,14,3],
            [1,3,5,3,1],
        ]
) / 234

# Approximation of 2nd derivation
EDGE_KERNEL = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# Sobel kernel: Smoothing & edge detection in one
SOBEL_X_KERNEL = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 4
SOBEL_Y_KERNEL = SOBEL_X_KERNEL.transpose()

def main():
    if len(sys.argv) != 4:
        usage()

    operation = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    if operation == 'resize':
        pixels = load_pixels(input_file)
        out_pixels = shrink(pixels, SHRINK_FACTOR, get_nearest_neighbour)

    elif operation == 'blur':
        pixels = load_pixels(input_file)
        out_pixels = convolute_2D(pixels, BLUR_KERNEL_LARGE)

    elif operation == 'edge':
        pixels = load_pixels(input_file, rgb=False)
        out_pixels = convolute_2D(pixels, EDGE_KERNEL)

    elif operation == 'sobel':
        pixels = load_pixels(input_file, rgb=False)

        # These will both be int16, with values (due to the nature of the
        # kernel) in (-2^{10}, 2^{10}). Squaring leads to values in (0,
        # 2^{20}), so we need a sufficiently large int.
        out_x = convolute_2D(pixels, SOBEL_X_KERNEL).astype(np.int32)
        out_y = convolute_2D(pixels, SOBEL_Y_KERNEL).astype(np.int32)
        out_pixels = np.sqrt(np.square(out_x) + np.square(out_y))

    else:
        usage()

    write_pixels(output_file, out_pixels)

def load_pixels(img_path, rgb=True):
    if rgb:
        img = cv2.imread(img_path)
        # Default is BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError("Unable to load image: {}".format(img_path))

    return img


def write_pixels(img_path, out_pixels):
    # Clamp to valid values
    out_pixels = out_pixels.clip(0, 255)
    # And ensure we write uint8s. OpenCV isn't too picky, but it seems a sane
    # thing to do.
    out_pixels = out_pixels.astype(np.uint8)

    if len(out_pixels.shape) == 3:
        # Must write BGR rather than RGB. Grayscale can be written directly
        out_pixels = cv2.cvtColor(out_pixels, cv2.COLOR_RGB2BGR)

    cv2.imwrite(img_path, out_pixels)

def convolute_2D(pixels, kernel):
    input_height = pixels.shape[0]
    input_width = pixels.shape[1]

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    # We need the ability to be able to subtract numbers (for eg edge
    # detection) without underflowing, so will use a signed integer and clamp
    # to [0, 255] later when writing
    output = np.zeros(pixels.shape, dtype=np.int16)

    # With an nxn kernel, the middle field is for the pixel itself, so we need
    # floor(n/2) padding on each side.
    pad_width = max(int(kernel_width / 2), int(kernel_height / 2))
    padded = edge_pad_2D(pixels, pad_width)

    # TODO should be able to use np.fromfunction or similar here
    for y in range(0, input_height):
        for x in range(0, input_width):
            # Corresponding pixel in padded image is offset by `pad_width` in
            # either direction. But as we want a window *centered* on the given
            # pixel, we subtract `pad_width` again, so it cancels out.
            window = padded[y : y + kernel_height, x : x + kernel_width]

            if len(pixels.shape) == 3:
                # Treat R, G, B separately
                for ch in range(3):
                    # We want component-wise multiplication of kernel and
                    # (sub)-matrix of the image, so use np.multiply, then sum
                    # the entries up.
                    output[y, x, ch] = np.multiply(kernel, window[:, :, ch]).sum()
            else:
                output[y, x] = np.multiply(kernel, window).sum()

    return output

# Takes a (height, width, 3) array representing an RGB image, and
# two-dimensionally edge-pads it along the height and width axes.
#
# Returns a (height + pad_width, width + pad_width, 3) array.
#
# Bootleg version of np.pad(ary, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), dtype=int)
# Probably less performant, but hey.
def edge_pad_2D(ary, pad_width):
    if pad_width == 0:
        return ary

    height = ary.shape[0]
    width = ary.shape[1]

    # Initiate zero-filled array with pad_width additional rows / columns on
    # every side.
    # 2D input => 2D output :)
    if len(ary.shape) == 2:
        out = np.zeros((height + 2 * pad_width, width + 2 * pad_width), dtype=int)
    else:
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
    print ("Usage: process.py <resize|blur|edge|sobel> <input_file> <output_file>")
    sys.exit(1)

if __name__ == '__main__':
    main()
