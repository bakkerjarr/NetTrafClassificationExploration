from classifiers import iscx_result_calc as rc
from sklearn.naive_bayes import GaussianNB
from numpy import float32 as np_float
import numpy.core.multiarray as np_array

__author__ = "Jarrod N. Bakker"


class NaiveBayesCls:

    def __init__(self, data_loader):
        """Initialise.

        :param data_loader: Object from where the data is fetched from.
        """
        self._data, self._labels = data_loader.get_data()
        #self._train_indices = data_loader.get_train_indices()
        #self._test_indices = data_loader.get_test_indices()
        self._kfold = data_loader.get_kfold()
        self._classifier = GaussianNB()

    def classify(self):
        """Classify DDoS flows using Naive Bayes.

        The data passed through to the fit() method cannot be a string
        type.

        :return: Something...
        """
        all_results = []  # Results from all fold trials
        fold_num = 1
        for train, test in self._kfold:
            print("Training Naive Bayes...")
            train_array = np_array.array(map(self._data.__getitem__,
                                             train)).astype(np_float)

            train_label_array = np_array.array(map(
                self._labels.__getitem__, train)).astype(np_float)
            self._classifier.fit(train_array, train_label_array)
            print("Testing classifier...")
            test_array = np_array.array(map(self._data.__getitem__,
                                             test)).astype(np_float)
            test_label_array = np_array.array(map(
                self._labels.__getitem__, test)).astype(np_float)
            test_size = len(test)
            pred = self._classifier.predict(test_array)
            mislabeled = (test_label_array != pred).sum()
            tp, tn, fp, fn = rc.calculate_tpn_fpn(test_label_array, pred)
            #print("TP: {0}\tTN: {1}\tFP: {2}\tFN: {3}".format(tp, tn,
            #                                                  fp, fn))
            detection_rate = rc.detection_rate(tp, fn)
            false_pos_rate = rc.false_positive_rate(tn, fp)
            #print("Detection rate: {0}\tFalse positive rate: "
            #      "{1}".format(detection_rate, false_pos_rate))
            #print("Number of mislabelled points out of a total {0} "
            #      "points : {1}".format(test_size, mislabeled))
            all_results.append([fold_num, tp, tn, fp, fn, detection_rate,
                                false_pos_rate, mislabeled, test_size])
            fold_num += 1
        return all_results
