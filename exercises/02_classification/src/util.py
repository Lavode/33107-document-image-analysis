from pathlib import Path

import cv2

DATA_ROOT = '../mnist'
TEST_DIR = 'test'
TRAIN_DIR = 'train'
VALIDATION_DIR = 'val'

DEBUG=True

def debug(msg):
    if DEBUG:
        print(msg)

# Loads image using CV2, returns a n*m*3 (for RGB) respectively a n*m (for
# grayscale) numpy array of pixel values.
def load_img(img_path, rgb=True):
    if rgb:
        img = cv2.imread(img_path)
        # Default is BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError("Unable to load image: {}".format(img_path))

    return img

# Generator yielding test datasets and their corresponding class
def _train_data(classes, limit=None):
    for cls in classes:
        debug("Loading training data for class: {}".format(cls))
        path = Path(DATA_ROOT, TRAIN_DIR, cls)

        for img in _images_from_directory(path, limit):
            yield (img, cls)

# Generator yielding training data
def _test_data(classes, limit=None):
    for cls in classes:
        debug("Loading testing data for class: {}".format(cls))
        path = Path(DATA_ROOT, TEST_DIR, cls)

        for img in _images_from_directory(path, limit):
            yield img, cls

# Generator yielding grayscale images (as numpy arrays) from the specified directory.
def _images_from_directory(path, limit=None):
    if not path.exists():
        raise RuntimeError("Path does not exist: {}".format(path))

    count = 0

    for img_path in path.glob('*.png'):
        # MNIST database is grayscale
        img_pixels = load_img(str(img_path), rgb=False)
        yield img_pixels
        count += 1

        if limit is not None and count >= limit:
            break
