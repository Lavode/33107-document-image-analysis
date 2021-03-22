import time

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

import classifier
import features as f
import util

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

# Limit of samples we'll allow for training. Required to preallocate arrays for
# sample storage.
TRAINING_SAMPLES_LIMIT = 60000
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
                'f': f.extract_flattened_pixels, # No real extraction required, comparison will operate on pixel values directly
                'f_compare': f.compare_euclidean_distance,
                'dim': (784,), # 28*28 pixels, flattened
                'type': np.uint8,
            },
    ]

    # _custom_classifier(features)
    _scikit_classifiers(features)

# Evaluate various Scikit classifiers with MNIST data
def _scikit_classifiers(features):
    # - Fix random seed
    # - Increase iterations such that it converges
    # - If n_samples > n_features => dual = False
    # clsf = LinearSVC(random_state=42, max_iter=10000, dual=False)
    # classifier_type = 'LinearSVC'

    # - Set neighbour count for nn search
    # clsf = KNeighborsClassifier(n_neighbors=5)
    # classifier_type = 'KNeighbors[k=5]'

    # - Fix random seed
    clsf = MLPClassifier(random_state=42)
    classifier_type = 'MLP'

    extraction_time, training_time, sample_count = _train_scikit_classifier(clsf, features)

    stats, prediction_time = _test_scikit_classifier(clsf, features)
    for cls in sorted(stats.keys()):
        util.debug("Class {} => {}% correct ({} total)".format(cls, round(stats[cls]['accuracy'] * 100, 1), stats[cls]['total']))


    print("")

    feature_id = ','.join([ d['name'] for d in features ])

    # And some machine-readable output suitable for appending to a CSV for
    # analysis
    print("classifier_type,features,extraction_time,training_time,prediction_time,training_sample_count")
    print("{},\"{}\",{},{},{},{}".format(classifier_type, feature_id, extraction_time, training_time, prediction_time, sample_count))

    print("")

    print("classifier_type,features,class,total,accuracy")
    for cls in stats:
        print("{},\"{}\",{},{},{}".format(classifier_type, feature_id, cls, stats[cls]['total'], stats[cls]['accuracy']))


# Train given Scikit classifier
def _train_scikit_classifier(clsf, features):
    start = time.time()
    x, y = _get_scikit_training_data(features)
    extraction_time = time.time() - start

    print("Dimensions of X: {}".format(x.shape))
    print("Dimensions of Y: {}".format(y.shape))

    print("Training classifier")
    start = time.time()
    clsf.fit(x, y)
    training_time = time.time() - start

    return extraction_time, training_time, x.shape[0]

# Test accuracy of given Scikit classifier
def _test_scikit_classifier(clsf, features):
    print("Testing classifier")
    xtest, ytest = _get_scikit_testing_data(features)

    start = time.time()
    predictions = clsf.predict(xtest)
    prediction_time = time.time() - start

    stats = {}
    for cls in CLASSES:
        stats[int(cls)] = { 'total': 0, 'correct': 0, 'incorrect': 0 }

    idx = 0
    for prediction in predictions:
        prediction = int(prediction)
        actual = int(ytest[idx])

        stats[actual]['total'] += 1
        if prediction == actual:
            stats[actual]['correct'] += 1
        else:
            stats[actual]['incorrect'] += 1

        idx += 1

    for cls in stats:
        stats[cls]['accuracy'] = stats[cls]['correct'] / stats[cls]['total']

    return stats, prediction_time

# Get testing data suitable for use with Scikit.
def _get_scikit_testing_data(features):
    return _get_scikit_data(features, TRAINING_SAMPLES_LIMIT, util._test_data, [CLASSES])

# Get training data suitable for use with Scikit
def _get_scikit_training_data(features):
    return _get_scikit_data(features, TRAINING_SAMPLES_LIMIT, util._train_data, [CLASSES, MAX_SAMPLE_COUNT_PER_CLASS])

# Retrieves data from data-providing function, then preprocesses & stores it in
# a way that it can be used with Scikit classifiers.
#
# This involves:
# - Storing as X = (n_sample, n_feature), Y = (n_feature) sample and label vectors
# - Normalizing sample data
def _get_scikit_data(features, sample_limit, get_data_func, get_data_args):
    # Length of feature vector
    feature_length =_scikit_feature_length(features)

    class_count = len(CLASSES)

    x = np.zeros((sample_limit, feature_length))
    y = np.zeros(sample_limit)

    sample_idx = 0

    for img_pixels, cls in get_data_func(*get_data_args):
        x[sample_idx] = _scikit_features(img_pixels, features, feature_length)
        y[sample_idx] = int(cls)

        sample_idx += 1

    # sample_idx is now the *count* of actual samples, so we can use it to truncate x and y
    x = x[ : sample_idx]
    y = y[ : sample_idx]

    # Finally we'll normalize mean and variance
    print("Normalizing data")
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    return x, y

# Get 1-dimensional feature vector suitable for use with scikit for given image
def _scikit_features(img_pixels, features, feature_vector_length):
    out = np.zeros(feature_vector_length)
    feature_idx = 0

    for feature in features:
        feature_length = feature['dim'][0]
        out[feature_idx : feature_idx + feature_length] = feature['f'](img_pixels)
        feature_idx += feature_length

    return out

# Calculate length of (one-dimensional) feature vector to use for Scikit
# classifiers.
def _scikit_feature_length(features):
    length = 0

    for feature in features:
        dim = feature['dim']
        if len(dim) != 1:
            raise RuntimeError("Invalid feature dimensions {}, Scikit classifiers require 1-dimensional feature vector.".format(dim))
        length += dim[0]

    return length


# Evaluate own classifier with MNIST data
def _custom_classifier(features):
    clsf = classifier.Classifier(CLASSES, TRAINING_SAMPLES_LIMIT, features, 'avg')

    _train_classifier_from_data(clsf)
    clsf.finalize()

    stats = _test_custom_classifier(clsf)
    for cls in sorted(stats.keys()):
        util.debug("Class {} => {}% correct ({} total)".format(cls, round(stats[cls]['accuracy'] * 100, 1), stats[cls]['total']))

    # And some machine-readable output suitable for appending to a CSV for
    # analysis
    feature_id = ','.join([ d['name'] for d in features ])
    print("decision_mode,features,class,total,accuracy")
    for cls in stats:
        print("{},\"{}\",{},{},{}".format(clsf.decision_mode, feature_id, cls, stats[cls]['total'], stats[cls]['accuracy']))


# Test custom classifier with data from MNIST test directory
def _test_custom_classifier(classifier):
    util.debug("Testing classifier with test data")
    stats = {}
    for cls in CLASSES:
        stats[cls] = { 'total': 0, 'correct': 0, 'incorrect': 0 }

    # We'll throw away the class it yields
    for img_pixels, cls in util._test_data(CLASSES):
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
    for img_pixels, cls in util._train_data(CLASSES, MAX_SAMPLE_COUNT_PER_CLASS):
        classifier.train_img(img_pixels, cls)

if __name__ == '__main__':
    main()
