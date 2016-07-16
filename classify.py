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
from contextlib import contextmanager
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
        with open("test_time.txt", mode="a") as file_out:
            cur_dt = str(datetime.datetime.now())
            file_out.write("{0}\tTest started\n".format(cur_dt))

        csv_headings = "classifier, features, seed, trial_num, " \
                       "fold_num, TP, TN, FP, FN, TP_rate, FP_rate, " \
                       "num_mis, total_test\n"
        classifiers = [NaiveBayesCls, SVMCls, LDACls, QDACls,
                       DecisionTreeCls, RandomForestCls]
        num_trials = 10
        num_folds = 30

        if not self._iscx2012_loader.load_data():
            print("Failed to read data from file.")
            sys.exit(-1)

        features_set, labels = self._iscx2012_loader.get_data()

        with open("test_time.txt", mode="a") as file_out:
            cur_dt = str(datetime.datetime.now())
            file_out.write("{0}\t\tTesting classifiers: ".format(cur_dt))
            for i in range(len(classifiers)):
                if i != len(classifiers)-1:
                    file_out.write("{0}, ".format(classifiers[i].NAME))
                else:
                    file_out.write("{0}\n".format(classifiers[i].NAME))
            cur_dt = str(datetime.datetime.now())
            file_out.write("{0}\t\tFeature sets: ".format(cur_dt))
            fs_names = features_set.keys()
            for i in range(len(fs_names)):
                if i != len(fs_names)-1:
                    file_out.write("{0}, ".format(fs_names[i]))
                else:
                    file_out.write("{0}\n".format(fs_names[i]))

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
                for trial_num in range(1, num_trials+1):
                    skf = self._iscx2012_loader.get_kfold(num_folds,
                                                          seed)
                    # create the classifier, pass the data through
                    # call classify
                    results = cls(features_set[features],
                                  labels, skf).classify()
                    print("\tWriting results for trial {0}.".format(
                        trial_num))
                    try:
                        with open("test_time.txt", mode="a") as file_out:
                            cur_dt = str(datetime.datetime.now())
                            file_out.write("{0}\t\tWriting "
                                           "test results to "
                                           "file: {1}\n".format(
                                            cur_dt, file_name))
                        file_out = open(file_name, mode="a")
                        for r in results:
                            line = "{0}, {1}, {2}, {3}, {4}\n".format(
                                cls.NAME, features, seed, trial_num,
                                str(r)[1:-1])
                            file_out.write(line)
                    except IOError as err:
                        print("IOError writing results to file: "
                              "{0}".format(err))
                        with open("test_time.txt", mode="a") as err_out:
                            cur_dt = str(datetime.datetime.now())
                            err_out.write("{0}\t\tIOError writing "
                                          "results to file: "
                                          "{1}\n".format(cur_dt, err))
                    seed += 1

        with open("test_time.txt", mode="a") as file_out:
            cur_dt = str(datetime.datetime.now())
            file_out.write("{0}\t Test finished\n".format(cur_dt))
        print("TEST COMPLETE: Exiting...")

@contextmanager
def opened_w_error(filename, mode="r"):
    """A useful helper function for handling exceptions with the
    'with' statement.
    Taken from http://stackoverflow.com/questions/713794/catching-an
    -exception-while-using-a-python-with-statement/6090497#6090497.

    :param filename:
    :param mode:
    :return:
    """
    try:
        f = open(filename, mode)
    except IOError, err:
        yield None, err
    else:
        try:
            yield f, None
        finally:
            f.close()

if __name__ == "__main__":
    files = ["TestbedTueJun15-1Flows.xml",
             "TestbedTueJun15-2Flows.xml",
             "TestbedTueJun15-3Flows.xml"]
    c = Classify(files)
    c.run_tests()
