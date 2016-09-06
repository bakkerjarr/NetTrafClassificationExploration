# Copyright 2016 Jarrod N. Bakker
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numpy import float32 as np_float

import numpy.core.multiarray as np_array
from sklearn.naive_bayes import GaussianNB

from priliminary.classifiers import iscx_result_calc as rc

__author__ = "Jarrod N. Bakker"


class NaiveBayesCls:

    NAME = "Naive_Bayes"

    def __init__(self, config, data, labels, skf):
        """Initialise.

        :param config: Dict of config information for classifiers.
        :param data: Data set for the classifier to use.
        :param labels: Labels indicating if a flow is normal or attack.
        :param skf: StratifiedKFold object representing what data set
        elements belong in each fold.
        """
        self._config = None  # sklearn's Naive Bayes has no parameters
        self._data = data
        self._labels = labels
        self._kfold = skf

    def classify(self):
        """Classify DDoS flows using Naive Bayes.

        The data passed through to the fit() method cannot be a string
        type.

        :return: Results of the classification.
        """
        classifier = GaussianNB()
        all_results = []  # Results from all fold trials
        fold_num = 1
        for train, test in self._kfold:
            print("\tTraining Naive Bayes...")
            # NOTE: I have switched the training and testing set around.
            train_array = np_array.array(map(self._data.__getitem__,
                                             test)).astype(np_float)
            train_label_array = np_array.array(map(
                self._labels.__getitem__, test)).astype(np_float)
            classifier.fit(train_array, train_label_array)
            print("\tTesting classifier...")
            test_array = np_array.array(map(self._data.__getitem__,
                                             train)).astype(np_float)
            test_label_array = np_array.array(map(
                self._labels.__getitem__, train)).astype(np_float)
            test_size = len(train)  # Remember the switch of sets!
            pred = classifier.predict(test_array)
            mislabeled = (test_label_array != pred).sum()
            tp, tn, fp, fn = rc.calculate_tpn_fpn(test_label_array, pred)
            detection_rate = rc.detection_rate(tp, fn)
            false_pos_rate = rc.false_positive_rate(tn, fp)
            all_results.append([fold_num, tp, tn, fp, fn, detection_rate,
                                false_pos_rate, mislabeled, test_size])
            fold_num += 1
        return all_results
