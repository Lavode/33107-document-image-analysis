from pathlib import Path

import cv2

import numpy as np

DATA_ROOT = 'mnist'
TEST_DIR = 'test'
TRAIN_DIR = 'train'
VALIDATION_DIR = 'val'

CLASSES = set(
        [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
        ]
)

# Upper limit to preallocate sample arrays. Will be truncated in the end.
TRAINING_DATA_COUNT = 6000
MAX_SAMPLE_COUNT_PER_CLASS = 10

FEATURE_COUNT = 2
HORIZONTAL_PROFILE_IDX = 0
VERTICAL_PROFILE_IDX = 1

CLASSES_COUNT = len(CLASSES)

class Classifier:
    feature_count = 2

    def __init__(self, classes, max_sample_count, features):
        self.classes = classes
        self.max_sample_count = max_sample_count

        self.features = features

        # class => feature id => np array
        self.samples = {}
        for cls in classes:
            print("Initializing features of class {}".format(cls))
            self.samples[cls] = {}
            for feature in features:
                dim = (max_sample_count, ) + feature['dim']
                print("\tInitializing feature '{}' with dimension {}".format(feature['name'], dim))
                self.samples[cls][feature['id']] = np.zeros(dim)

        # class => feature id => np array
        self.representatives = {}
        for cls in classes:
            self.representatives[cls] = {}
            for feature in features:
                self.representatives[cls][feature['id']] = np.zeros(feature['dim'])

        # class => Maximum index of currently unused place in corresponding sample np array
        self.current_idx = {}
        for cls in classes:
            self.current_idx[cls] = 0

    def train_img(self, pixels, cls):
        if pixels.shape != (28, 28):
            raise RuntimeError("Only grayscale MNIT images supported. Cannot work with arrays of shape {}".format(pixels.shape))

        idx = self.current_idx[cls]

        for feature in self.features:
            self.samples[cls][feature['id']][idx] = feature['f'](pixels)

        self.current_idx[cls] += 1

    def finalize(self):
        print("Finalizing classifier")

        for cls in self.classes:
            print("Truncating unused samples")
            allocated = self.max_sample_count
            used = self.current_idx[cls]

            for feature in self.features:
                feature_id = feature['id']

                print("\tClass {} has {} samples allocated, {} used. Truncating...".format(cls, allocated, used))
                # `used` is highest *unused* index, so is also the number of
                # trained samples - hence the slicing just works
                self.samples[cls][feature_id] = self.samples[cls][feature_id][: used, :]

                print("Calculating representative as mean of used features")
                # Sum along sample axis (0), then divide by number of samples
                self.representatives[cls][feature_id] = (self.samples[cls][feature_id].sum(0) / used).astype(np.uint8)



def main():
    features = [
            {
                'id': 'horizontal_profile',
                'name': "Horizontal profile",
                'f': extract_horizontal_profile,
                'dim': (28,)
            },
            {
                'id': 'vertical_profile',
                'name': "Vertical profile",
                'f': extract_vertical_profile,
                'dim': (28,)
            },
    ]
    classifier = Classifier(CLASSES, TRAINING_DATA_COUNT, features)

    # pixels = load_img('mnist/train/0/21273.png', rgb=False)
    # classifier.train_img(pixels, '0')

    train_classifier_from_data(classifier)
    classifier.finalize()

    print(classifier.representatives['3']['vertical_profile'])

def train_classifier_from_data(classifier):
    for cls in CLASSES:
        cnt = 0
        print("Training for class: {}".format(cls))

        path = Path(DATA_ROOT, TRAIN_DIR, cls)
        if not path.exists():
            raise RuntimeError("Training path does not exist: {}".format(path))

        for img_path in path.glob('*.png'):
            # MNIST database is grayscale
            img_pixels = load_img(str(img_path), rgb=False)
            classifier.train_img(img_pixels, cls)
            cnt += 1

            if cnt >= MAX_SAMPLE_COUNT_PER_CLASS:
                break

def extract_horizontal_profile(pixels):
    return profile_along_axis(pixels, 0)

def extract_vertical_profile(pixels):
    return profile_along_axis(pixels, 1)

def profile_along_axis(pixels, axis):
    # Data we get is grayscale so in [0, 256). To ensure the profile is in
    # this range too, we normalize by dividing by the shape along this
    # axis.
    return (pixels / pixels.shape[axis]).sum(axis).astype(np.uint8)


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

if __name__ == '__main__':
    main()
