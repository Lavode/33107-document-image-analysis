import numpy as np

def extract_horizontal_profile(pixels):
    return extract_profile_along_axis(pixels, 0)

def extract_vertical_profile(pixels):
    return extract_profile_along_axis(pixels, 1)

def extract_profile_along_axis(pixels, axis):
    # Data we get is grayscale so in [0, 256). To ensure the profile is in
    # this range too, we normalize by dividing by the shape along this
    # axis.
    return (pixels / pixels.shape[axis]).sum(axis).astype(np.uint8)

def compare_profiles(a, b):
    # We'll use the sum of absolute component-wise differences, normalized to [0, 256)
    # Casting to a signed & sufficiently large type first
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32)).sum() / a.shape[0]

    # Finally normalize to [0, 1)
    return diff / 256.0

def extract_flattened_pixels(pixels):
    return pixels.flatten()

def compare_euclidean_distance(a, b):
    # Square of per-pixel distance, normalized to [0, 1) (sum normalized, then squared)
    per_pixel_diff = np.square((a.astype(np.int32) - b.astype(np.int32)) / 256)

    # And the sum thereof, normalized to [0, 1) once more
    sum_of_diff = np.sum(per_pixel_diff) / (28*28)

    # And finally the sqrt
    return np.sqrt(sum_of_diff)

def extract_average_pixel(pixels):
    # With integers you risk having two class representatives have the same
    # average value, in which case results might be non-deterministic at worst,
    # and nonsensical at best.
    return float(pixels.sum()) / (pixels.shape[0] * pixels.shape[1])

def compare_average_pixel(a, b):
    # Average pixel value will be in [0, 255], normalize to [0, 1]
    return abs(a - b) / 255
