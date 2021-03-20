from pathlib import Path

import cv2

import numpy as np

DATA_ROOT = 'mnist'
TEST_DIR = 'test'
TRAIN_DIR = 'train'
VALIDATION_DIR = 'val'

DEBUG=True

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
# Optional limit on how much training data to consume *per class*. Set to
# `None` to disable.
MAX_SAMPLE_COUNT_PER_CLASS = None

class Classifier:
    def __init__(self, classes, max_sample_count, features, decision_mode):
        self.classes = classes
        self.max_sample_count = max_sample_count

        self.features = features

        self.decision_mode = decision_mode

        # class => feature id => np array
        self.samples = {}
        for cls in classes:
            debug("Initializing features of class {}".format(cls))
            self.samples[cls] = {}
            for feature in features:
                dim = (max_sample_count, ) + feature['dim']
                debug("\tInitializing feature '{}' with dimension {}".format(feature['name'], dim))
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
        debug("Finalizing classifier")

        for cls in self.classes:
            debug("Processing class {}".format(cls))
            allocated = self.max_sample_count
            used = self.current_idx[cls]

            for feature in self.features:
                debug("\tProcessing feature {}".format(feature['name']))
                feature_id = feature['id']

                debug("\t\t{} samples allocated, {} used. Truncating...".format(allocated, used))
                # `used` is highest *unused* index, so is also the number of
                # trained samples - hence the slicing just works
                self.samples[cls][feature_id] = self.samples[cls][feature_id][: used, :]

                debug("\t\tCalculating representative as mean of used features")
                # Sum along sample axis (0), then divide by number of samples
                self.representatives[cls][feature_id] = (self.samples[cls][feature_id].sum(0) / used).astype(np.uint8)

    def test(self, pixels):
        # class => feature => distance
        # Might look like:
        # { '0' => { 'euclidean' => 0.2, 'horizontal_profile' => 0.4 }, '1' => { 'euclidean' => 0.3, 'horizontal_profile' => 0.5 } }
        distances = {}

        # Populate dict with the 'distance' metric between every class and
        # the test sample, by every feature
        for cls in self.classes:
            distances[cls] = {}
            representative = self.representatives[cls]

            for feature in self.features:
                feature_id = feature['id']

                test_feature = feature['f'](pixels)
                dist = feature['f_compare'](test_feature, representative[feature_id])
                distances[cls][feature_id] = dist


        feature_count = len(self.features)

        # And calculate a few aggregations:
        # - sum of features' distances per class
        # - average distance per class
        # - minimal distance per class
        for cls in self.classes:
            distances[cls]['_sum'] = 0
            distances[cls]['_min'] = 1
            for feature in self.features:
                feature_id = feature['id']
                feature_distance = distances[cls][feature_id]

                distances[cls]['_sum'] += feature_distance
                if feature_distance < distances[cls]['_min']:
                    distances[cls]['_min'] = feature_distance

            distances[cls]['_avg'] = distances[cls]['_sum'] / feature_count

        if self.decision_mode == 'avg':
            sorted_dists = sorted(distances.items(), key=lambda x: x[1]['_avg'])
        elif self.decision_mode == 'min':
            sorted_dists = sorted(distances.items(), key=lambda x: x[1]['_min'])
        else:
            raise RuntimeError("Invalid decision mode: {}".format(decision_mode))


        return sorted_dists[0]


def main():
    features = [
            {
                'id': 'average_pixel',
                'name': "Average pixel",
                'f': extract_average_pixel,
                'f_compare': compare_average_pixel,
                'dim': (1,)
            },
            # {
            #     'id': 'horizontal_profile',
            #     'name': "Horizontal profile",
            #     'f': extract_horizontal_profile,
            #     'f_compare': compare_profiles,
            #     'dim': (28,)
            # },
            # {
            #     'id': 'vertical_profile',
            #     'name': "Vertical profile",
            #     'f': extract_vertical_profile,
            #     'f_compare': compare_profiles,
            #     'dim': (28,)
            # },
            # {
            #     'id': 'euclidean_distance',
            #     'name': 'Euclidean distance',
            #     'f': extract_flattened_pixels, # No real extraction requried, comparison will operate on pixel values directly
            #     'f_compare': compare_euclidean_distance,
            #     'dim': (784,), # 28*28 pixels, flattened
            # },
    ]
    classifier = Classifier(CLASSES, TRAINING_DATA_COUNT, features, 'min')

    train_classifier_from_data(classifier)
    classifier.finalize()

    stats = test_classifier(classifier)
    for cls in sorted(stats.keys()):
        debug("Class {} => {}% correct".format(cls, round(stats[cls]['accuracy'] * 100, 1)))

    # And some machine-readable output suitable for appending to a CSV for
    # analysis
    feature_id = ','.join([ d['name'] for d in features ])
    print("decision_mode,features,class,total,accuracy")
    for cls in stats:
        print("{},\"{}\",{},{},{}".format(classifier.decision_mode, feature_id, cls, stats[cls]['total'], stats[cls]['accuracy']))

def test_classifier(classifier):
    debug("Testing classifier with test data")
    stats = {}
    for cls in CLASSES:
        stats[cls] = { 'total': 0, 'correct': 0, 'incorrect': 0 }

        path = Path(DATA_ROOT, TEST_DIR, cls)
        if not path.exists():
            raise RuntimeError("Testing path does not exist: {}".format(path))

        for img_path in path.glob('*.png'):
            # MNIST database is grayscale
            img_pixels = load_img(str(img_path), rgb=False)

            classification, score = classifier.test(img_pixels)
            # debug("Actual: {}, Classification: {}, Score: {}".format(cls, classification, score))

            stats[cls]['total'] += 1
            if classification == cls:
                stats[cls]['correct'] += 1
            else:
                stats[cls]['incorrect'] += 1

    for cls in stats:
        stats[cls]['accuracy'] = stats[cls]['correct'] / stats[cls]['total']

    return stats


def train_classifier_from_data(classifier):
    for cls in CLASSES:
        cnt = 0
        debug("Training for class: {}".format(cls))

        path = Path(DATA_ROOT, TRAIN_DIR, cls)
        if not path.exists():
            raise RuntimeError("Training path does not exist: {}".format(path))

        for img_path in path.glob('*.png'):
            # MNIST database is grayscale
            img_pixels = load_img(str(img_path), rgb=False)
            classifier.train_img(img_pixels, cls)
            cnt += 1

            if MAX_SAMPLE_COUNT_PER_CLASS is not None and cnt >= MAX_SAMPLE_COUNT_PER_CLASS:
                break

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
    # Square of per-pixel distance, normalized to [0, 1)
    per_pixel_diff = np.square((a.astype(np.int32) - b.astype(np.int32)) / 256)

    # And the sum thereof, normalized to [0, 1) once more
    sum_of_diff = np.sum(per_pixel_diff) / (28*28)

    # And finally the sqrt
    return np.sqrt(sum_of_diff)

def extract_average_pixel(pixels):
    return pixels.sum() / (pixels.shape[0] * pixels.shape[1])

def compare_average_pixel(a, b):
    # Average pixel value will be in [0, 255], normalize to [0, 1]
    return abs(a - b) / 255


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

def debug(msg):
    if DEBUG:
        print(msg)

if __name__ == '__main__':
    main()
