from classifiers import iscx_result_calc as rc
from sklearn import svm
from numpy import float32 as np_float
import numpy.core.multiarray as np_array

__author__ = "Jarrod N. Bakker"


class SVMCls:

    def __init__(self, data_loader):
        """Initialise.

        :param data_loader: Object from where the data is fetched from.
        """
        self._test_data, self._test_labels = data_loader.get_test_data()
        self._train_data, self._train_labels = \
            data_loader.get_train_data()
        self._classifier = svm.SVC()  # TODO Try LinearSVC

    def classify(self):
        """Classify DDoS flows using Naive Bayes.

        The data passed through to the fit() method cannot be a string
        type.

        :return: Something...
        """
        print("Training SVM...")
        train_array = np_array.array(self._train_data).astype(np_float)
        train_label_array = np_array.array(self._train_labels).astype(
            np_float)
        self._classifier.fit(train_array, train_label_array)
        print("Testing classifier...")
        test_array = np_array.array(self._test_data).astype(np_float)
        test_label_array = np_array.array(self._test_labels).astype(
            np_float)
        test_size = len(self._test_data)
        pred = self._classifier.predict(test_array)
        mislabeled = (test_label_array != pred).sum()
        tp, tn, fp, fn = rc.calculate_tpn_fpn(test_label_array, pred)
        print("TP: {0}\tTN: {1}\tFP: {2}\tFN: {3}".format(tp, tn, fp,
                                                          fn))
        detection_rate = rc.detection_rate(tp, fn)
        false_pos_rate = rc.false_positive_rate(tn, fp)
        print("Detection rate: {0}\tFalse positive rate: {1}".format(
            detection_rate, false_pos_rate))
        print("Number of mislabeled points out of a total {0} points : "
              "{1}".format(test_size, mislabeled))
