from pathlib import Path

import numpy as np

import classifier
import features as f
import util

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


def main():
    features = [
            # {
            #     'id': 'average_pixel',
            #     'name': "Average pixel",
            #     'f': f.extract_average_pixel,
            #     'f_compare': f.compare_average_pixel,
            #     'dim': (1,),
            #     'type': np.float64,
            # },
            {
                'id': 'horizontal_profile',
                'name': "Horizontal profile",
                'f': f.extract_horizontal_profile,
                'f_compare': f.compare_profiles,
                'dim': (28,),
                'type': np.uint8,
            },
            {
                'id': 'vertical_profile',
                'name': "Vertical profile",
                'f': f.extract_vertical_profile,
                'f_compare': f.compare_profiles,
                'dim': (28,),
                'type': np.uint8,
            },
            {
                'id': 'euclidean_distance',
                'name': 'Euclidean distance',
                'f': f.extract_flattened_pixels, # No real extraction requried, comparison will operate on pixel values directly
                'f_compare': f.compare_euclidean_distance,
                'dim': (784,), # 28*28 pixels, flattened
                'type': np.uint8,
            },
    ]

    _own_classifier(features)
    # scikit_classifiers(features)

def scikit_classifiers():
    print("Hi")

# Train & test own classifier with MNIST data
def _own_classifier(features):
    clsf = classifier.Classifier(CLASSES, TRAINING_DATA_COUNT, features, 'avg')

    _train_classifier_from_data(clsf)
    clsf.finalize()

    stats = _test_classifier(clsf)
    for cls in sorted(stats.keys()):
        util.debug("Class {} => {}% correct ({} total)".format(cls, round(stats[cls]['accuracy'] * 100, 1), stats[cls]['total']))

    # And some machine-readable output suitable for appending to a CSV for
    # analysis
    feature_id = ','.join([ d['name'] for d in features ])
    print("decision_mode,features,class,total,accuracy")
    for cls in stats:
        print("{},\"{}\",{},{},{}".format(clsf.decision_mode, feature_id, cls, stats[cls]['total'], stats[cls]['accuracy']))


# Test custom classifier with data from MNIST test directory
def _test_classifier(classifier):
    util.debug("Testing classifier with test data")
    stats = {}
    for cls in CLASSES:
        stats[cls] = { 'total': 0, 'correct': 0, 'incorrect': 0 }

        path = Path(DATA_ROOT, TEST_DIR, cls)
        if not path.exists():
            raise RuntimeError("Testing path does not exist: {}".format(path))

        for img_path in path.glob('*.png'):
            # MNIST database is grayscale
            img_pixels = util.load_img(str(img_path), rgb=False)

            classification, score = classifier.test(img_pixels)
            # util.debug("Actual: {}, Classification: {}, Score: {}".format(cls, classification, score))

            stats[cls]['total'] += 1
            if classification == cls:
                stats[cls]['correct'] += 1
            else:
                stats[cls]['incorrect'] += 1

    for cls in stats:
        stats[cls]['accuracy'] = stats[cls]['correct'] / stats[cls]['total']

    return stats


# Train custom classifier with data from MNIST test directory
def _train_classifier_from_data(classifier):
    for cls in CLASSES:
        cnt = 0
        util.debug("Training for class: {}".format(cls))

        path = Path(DATA_ROOT, TRAIN_DIR, cls)
        if not path.exists():
            raise RuntimeError("Training path does not exist: {}".format(path))

        for img_path in path.glob('*.png'):
            # MNIST database is grayscale
            img_pixels = util.load_img(str(img_path), rgb=False)
            classifier.train_img(img_pixels, cls)
            cnt += 1

            if MAX_SAMPLE_COUNT_PER_CLASS is not None and cnt >= MAX_SAMPLE_COUNT_PER_CLASS:
                break



if __name__ == '__main__':
    main()
