import numpy as np

import util

class Classifier:
    def __init__(self, classes, max_sample_count, features, decision_mode):
        self.classes = classes
        self.max_sample_count = max_sample_count

        self.features = features

        self.decision_mode = decision_mode

        # class => feature id => np array
        self.samples = {}
        for cls in classes:
            util.debug("Initializing features of class {}".format(cls))
            self.samples[cls] = {}
            for feature in features:
                dim = (max_sample_count, ) + feature['dim']
                util.debug("\tInitializing feature '{}' with dimension {}".format(feature['name'], dim))
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
        util.debug("Finalizing classifier")

        for cls in self.classes:
            util.debug("Processing class {}".format(cls))
            allocated = self.max_sample_count
            used = self.current_idx[cls]

            for feature in self.features:
                util.debug("\tProcessing feature {}".format(feature['name']))
                feature_id = feature['id']

                util.debug("\t\t{} samples allocated, {} used. Truncating...".format(allocated, used))
                # `used` is highest *unused* index, so is also the number of
                # trained samples - hence the slicing just works
                self.samples[cls][feature_id] = self.samples[cls][feature_id][: used, :]

                util.debug("\t\tCalculating representative as mean of used features")
                # Sum along sample axis (0), then divide by number of samples
                self.representatives[cls][feature_id] = (self.samples[cls][feature_id].sum(0) / used).astype(feature['type'])

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

