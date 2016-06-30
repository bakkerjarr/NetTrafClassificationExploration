"""Load data from a data set then pass it to a classifier.
"""

from classifiers.iscx_naive_bayes import NaiveBayesCls
from classifiers.iscx_svm import SVMCls
from data.iscx_ids_2012 import ISCX2012IDS
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

    def test_nb(self):
        """Test a Naive Bayes classifier.

        :return:
        """
        csv_headings = "classifier, features, seed, trial_num, " \
                       "fold_num, TP, TN, FP, FN, TP_rate, FP_rate, " \
                       "num_mis, total_test\n"
        classifier = "Naive Bayes"
        features = '["log(totalSourceBytes)"; "flowDuration"]'
        seed = 99999999
        num_trials = 5
        num_folds = 10

        if not self._iscx2012_loader.load_data():
            print("Failed to read data from file.")
            sys.exit(-1)

        with open("naive_bayes_results.csv", mode="w") as file_out:
            file_out.write(csv_headings)
            for trial_num in range(1, num_trials + 1):
                self._iscx2012_loader.prepare_data(num_folds, seed)
                naive_bayes = NaiveBayesCls(self._iscx2012_loader)
                results = naive_bayes.classify()
                print("Writing results for trial {0}.".format(trial_num))
                for r in results:
                    line = "{0}, {1}, {2}, {3}, {4}\n".format(
                        classifier, features, seed, trial_num,
                        str(r)[1:-1])
                    file_out.write(line)
                seed += 1
        print("TEST COMPLETE: Exiting...")

if __name__ == "__main__":
    files = ["TestbedTueJun15-1Flows.xml",
             "TestbedTueJun15-2Flows.xml",
             "TestbedTueJun15-3Flows.xml"]
    c = Classify(files)
    c.test_nb()
