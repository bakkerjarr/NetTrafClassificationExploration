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

from config_loader import ConfigLoader
from classifiers.iscx_knn import KNNCls
from classifiers.iscx_naive_bayes import NaiveBayesCls
from classifiers.iscx_qda import QDACls
from classifiers.iscx_random_forest import RandomForestCls
from classifiers.iscx_svm_rbf import SVMCls
from data.iscx_ids_2012 import ISCX2012IDS

from os import path
import datetime
import sys

__author__ = "Jarrod N. Bakker"


class Classify:
    """Main class.
    """

    _CONFIG_DIR = "config"
    _TEST_DEBUG = "test_time.txt"
    _WORKING_DIR = path.dirname(__file__)

    def __init__(self, config_file_name, dataset_files):
        """Initialise the program.

        :param config_file_name: Name of the config file.
        :param dataset_files: List of dataset file names.
        """
        self._config_file_path = path.join(self._WORKING_DIR,
                                           self._CONFIG_DIR,
                                           config_file_name)
        self._config_loader = ConfigLoader(self._config_file_path)
        self._config_loader.read_config()
        self._dataset_files = dataset_files
        self._iscx2012_loader = ISCX2012IDS(dataset_files)

    def run_tests(self):
        """Test a bunch of classifiers.

        :return: ?
        """
        with open(self._TEST_DEBUG, mode="a") as f_debug:
            cur_dt = str(datetime.datetime.now())
            f_debug.write("{0}\tTest started\n".format(cur_dt))

        csv_headings = "classifier, features, seed, trial_num, " \
                       "fold_num, TP, TN, FP, FN, TP_rate, FP_rate, " \
                       "num_mis, total_test\n"
        classifiers = [KNNCls, NaiveBayesCls, QDACls, RandomForestCls,
                       SVMCls]
        num_trials = 10
        num_folds = 30

        if not self._iscx2012_loader.load_data():
            print("Failed to read data from file.")
            sys.exit(-1)

        features_set, labels = self._iscx2012_loader.get_data()

        with open(self._TEST_DEBUG, mode="a") as f_debug:
            cur_dt = str(datetime.datetime.now())
            f_debug.write("{0}\t\tTesting classifiers: ".format(cur_dt))
            for i in range(len(classifiers)):
                if i != len(classifiers)-1:
                    f_debug.write("{0}, ".format(classifiers[i].NAME))
                else:
                    f_debug.write("{0}\n".format(classifiers[i].NAME))
            cur_dt = str(datetime.datetime.now())
            f_debug.write("{0}\t\tFeature sets: ".format(cur_dt))
            fs_names = features_set.keys()
            for i in range(len(fs_names)):
                if i != len(fs_names)-1:
                    f_debug.write("{0}, ".format(fs_names[i]))
                else:
                    f_debug.write("{0}\n".format(fs_names[i]))

        for features in features_set:
            for cls in classifiers:
                print("Testing features [{0}] with {1}.".format(
                    features, cls))
                seed = 99999999
                result_file = "{0}_{1}-fold_results.csv".format(
                    cls.NAME, num_folds)
                # If the results file does not exist we should create
                # one and write a header to it.
                if not path.isfile(result_file):
                    print("Creating file: {0}".format(result_file))
                    with open(result_file, mode="w") as f_results:
                        f_results.write(csv_headings)
                for trial_num in range(1, num_trials+1):
                    skf = self._iscx2012_loader.get_kfold(num_folds,
                                                          seed)
                    # create the classifier, pass the data through
                    # call classify
                    results = cls(
                        self._config_loader.get_classifier_config(),
                        features_set[features], labels, skf).classify()
                    print("\tWriting results for trial {0}.".format(
                        trial_num))
                    try:
                        with open(self._TEST_DEBUG, mode="a") as \
                                f_debug:
                            cur_dt = str(datetime.datetime.now())
                            f_debug.write("{0}\t\tWriting "
                                          "test results to "
                                          "file: {1}\tfeatures:{2}"
                                          "\ttrial: {3}\n".format(
                                           cur_dt, result_file,
                                           features, trial_num))
                        f_results = open(result_file, mode="a")
                        for r in results:
                            line = "{0}, {1}, {2}, {3}, {4}\n".format(
                                cls.NAME, features, seed, trial_num,
                                str(r)[1:-1])
                            f_results.write(line)
                    except IOError as err:
                        print("IOError writing results to file: "
                              "{0}".format(err))
                        with open(self._TEST_DEBUG, mode="a") as f_debug:
                            cur_dt = str(datetime.datetime.now())
                            f_debug.write("{0}\t\tIOError writing "
                                          "results to file: "
                                          "{1}\n".format(cur_dt, err))
                    finally:
                        f_results.close()
                    seed += 1

        with open(self._TEST_DEBUG, mode="a") as f_debug:
            cur_dt = str(datetime.datetime.now())
            f_debug.write("{0}\t Test finished\n".format(cur_dt))
        print("TEST COMPLETE: Exiting...")


if __name__ == "__main__":
    config_file_name = "classifiers.yaml"
    files = ["TestbedTueJun15-1Flows.xml",
             "TestbedTueJun15-2Flows.xml",
             "TestbedTueJun15-3Flows.xml"]
    c = Classify(config_file_name, files)
    c.run_tests()
