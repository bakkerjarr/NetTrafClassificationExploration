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

from priliminary.data.iscx_ids_2012 import TagValue

__author__ = "Jarrod N. Bakker"

"""Contains functions for interpreting the test output for the ISCX
IDS 2012 dataset.
"""


def calculate_tpn_fpn(test_labels, pred):
    """Calculate TP, TN, FP and FN.

    TP: True positives. TN: True negatives. FP: False positives.
    FN: False negatives.

    :param test_labels: Actual labels for the test set.
    :param pred: Predicted labels for the test set.
    :return: TP, TN, FP, FN as integers in a tuple.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    norm = TagValue.Normal
    for i in range(test_labels.size):
        if test_labels[i] == norm:
            if pred[i] == norm:
                tn += 1
            else:
                fp += 1
        else:
            if pred[i] == norm:
                fn += 1
            else:
                tp += 1
    return tp, tn, fp, fn


def calculate_tpn_fpn_anom(test_labels, pred):
    """Calculate TP, TN, FP and FN for anomaly detection context.

    The difference here is that different values are used for 'class'
    labels. 1 indicates inliers or normal, -1 indicates outliers or
    anomalous.

    TP: True positives. TN: True negatives. FP: False positives.
    FN: False negatives.

    :param test_labels: Actual labels for the test set.
    :param pred: Predicted labels for the test set.
    :return: TP, TN, FP, FN as integers in a tuple.
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    norm = 1  # an inlier
    for i in range(test_labels.size):
        if test_labels[i] == norm:
            if pred[i] == norm:
                tn += 1
            else:
                fp += 1
        else:
            if pred[i] == norm:
                fn += 1
            else:
                tp += 1
    return tp, tn, fp, fn


def detection_rate(tp, fn):
    """Calculate the detection rate.

    :param tp: Number of true positives.
    :param fn: Number of false negatives.
    :return: The detection rate.
    """
    if (tp+fn) == 0:
        return 0
    return tp/float(tp+fn)


def false_positive_rate(tn, fp):
    """Calculate the detection rate.

    :param tn: Number of true negatives.
    :param fp: Number of false positives.
    :return: The detection rate.
    """
    if (fp+tn) == 0:
        return 0
    return fp/float(fp+tn)
