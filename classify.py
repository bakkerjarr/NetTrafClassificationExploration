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

"""Load data from a data set then pass it to a classifier.
"""

from classifiers.iscx_naive_bayes import NaiveBayesCls
from classifiers.iscx_svm import SVMCls
from classifiers.iscx_lda import LDACls
from classifiers.iscx_qda import QDACls
from classifiers.iscx_decisiontree import DecisionTreeCls
from classifiers.iscx_random_forest import RandomForestCls
from data.iscx_ids_2012 import ISCX2012IDS
from os.path import isfile
import datetime
import sys

__author__ = "Jarrod N. Bakker"


class Classify:
    """Main class.
    """

    def __init__(self, dataset_files):
        """Initialise the program.

        :param dataset_files: List of dataset file names.
        """
        self._dataset_files = dataset_files
        self._iscx2012_loader = ISCX2012IDS(dataset_files)

    def run_tests(self):
        """Test a bunch of classifiers.

        :return: ?
        """
        csv_headings = "classifier, features, seed, trial_num, " \
                       "fold_num, TP, TN, FP, FN, TP_rate, FP_rate, " \
                       "num_mis, total_test\n"
        classifiers = [NaiveBayesCls, SVMCls, LDACls, QDACls,
                       DecisionTreeCls, RandomForestCls]
        num_trials = 1
        num_folds = 2

        if not self._iscx2012_loader.load_data():
            print("Failed to read data from file.")
            sys.exit(-1)

        features_set, labels = self._iscx2012_loader.get_data()
        for features in features_set:
            for cls in classifiers:
                print("Testing features [{0}] with {1}.".format(
                    features, cls))
                seed = 99999999
                file_name = "{0}_{1}-fold_results.csv".format(
                    cls.NAME, num_folds)
                # If the results file does not exist we should create
                # one and write a header to it.
                if not isfile(file_name):
                    print("Creating file: {0}".format(file_name))
                    with open(file_name, mode="w") as new_file:
                        new_file.write(csv_headings)
                with open(file_name, mode="a") as file_out:
                    for trial_num in range(1, num_trials+1):
                        skf = self._iscx2012_loader.get_kfold(num_folds,
                                                              seed)
                        # create the classifier, pass the data through
                        # call classify
                        results = cls(features_set[features],
                                      labels, skf).classify()
                        print("\tWriting results for trial {0}.".format(
                            trial_num))
                        for r in results:
                            line = "{0}, {1}, {2}, {3}, {4}\n".format(
                                cls.NAME, features, seed, trial_num,
                                str(r)[1:-1])
                            file_out.write(line)
                        seed += 1
        print("TEST COMPLETE: Exiting...")

if __name__ == "__main__":
    with open("test_time.txt", mode="a") as file_out:
        cur_dt = str(datetime.datetime.now())
        file_out.write("Test started at: {0}\n".format(cur_dt))
    files = ["TestbedTueJun15-1Flows.xml",
             "TestbedTueJun15-2Flows.xml",
             "TestbedTueJun15-3Flows.xml"]
    c = Classify(files)
    c.run_tests()
    with open("test_time.txt", mode="a") as file_out:
        cur_dt = str(datetime.datetime.now())
        file_out.write("Test finished at: {0}\n".format(cur_dt))
