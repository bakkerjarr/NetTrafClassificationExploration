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

from classifiers import iscx_result_calc as rc
from sklearn import svm
from numpy import float32 as np_float
import numpy.core.multiarray as np_array

__author__ = "Jarrod N. Bakker"


class SVMCls:

    NAME = "SVM"

    def __init__(self, data, labels, skf):
        """Initialise.

        :param data: Data set for the classifier to use.
        :param labels: Labels indicating if a flow is normal or attack.
        :param skf: StratifiedKFold object representing what data set
        elements belong in each fold.
        """
        self._data = data
        self._labels = labels
        self._kfold = skf
        self._classifier = svm.SVC()  # TODO Try LinearSVC

    def classify(self):
        """Classify DDoS flows using a Support Vector Machine.

        Note that SVM cannot handle too many data points for training.
        The exact number however is not currently known... Therefore use
        the StratifiedKFold object to obtain an even smaller training
        set. Alternatively, switch the training and testing sets around.
        It's an ugly hack...
        
        The data passed through to the fit() method cannot be a string
        type.

        :return: Results of the classification.
        """
        all_results = []  # Results from all fold trials
        fold_num = 1
        for train, test in self._kfold:
            print("\tTraining SVM...")
            # NOTE: I have switched the training and testing set around.
            train_array = np_array.array(map(self._data.__getitem__,
                                             test)).astype(np_float)
            train_label_array = np_array.array(map(
                self._labels.__getitem__, test)).astype(np_float)
            self._classifier.fit(train_array, train_label_array)
            print("\tTesting classifier...")
            test_array = np_array.array(map(self._data.__getitem__,
                                             train)).astype(np_float)
            test_label_array = np_array.array(map(
                self._labels.__getitem__, train)).astype(np_float)
            test_size = len(test)
            pred = self._classifier.predict(test_array)
            mislabeled = (test_label_array != pred).sum()
            tp, tn, fp, fn = rc.calculate_tpn_fpn(test_label_array, pred)
            # print("TP: {0}\tTN: {1}\tFP: {2}\tFN: {3}".format(tp, tn,
            #                                                   fp, fn))
            detection_rate = rc.detection_rate(tp, fn)
            false_pos_rate = rc.false_positive_rate(tn, fp)
            # print("Detection rate: {0}\tFalse positive rate: "
            #       "{1}".format(detection_rate, false_pos_rate))
            # print("Number of mislabelled points out of a total {0} "
            #       "points : {1}".format(test_size, mislabeled))
            all_results.append([fold_num, tp, tn, fp, fn, detection_rate,
                                false_pos_rate, mislabeled, test_size])
            fold_num += 1
        return all_results
