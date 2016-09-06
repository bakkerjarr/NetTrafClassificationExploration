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

import random
from lxml import etree

from sklearn.cross_validation import StratifiedKFold

from vetted.data import iscx_ids_2012_features as iscx_features

__author__ = "Jarrod N. Bakker"


class ISCX2012IDS:

    _BASE_PATH = "/vol/nerg-solar/bakkerjarr/Datasets/ISCXIDS2012" \
                 "/labeled_flows_xml/"

    def __init__(self, fnames):
        """Initialise.

        :param fnames: List of dataset file names.
        """
        self._rand = random
        self._dataset_files = []
        for f in fnames:
            self._dataset_files.append(self._BASE_PATH + f)
        self._raw_data = []
        self._labels = []
        self._data = []
        self._train_indices = []
        self._test_indices = []
        self._skf = None
        self._num_normal = 0
        self._num_attack = 0

    def load_data(self):
        """Load data from data sets, select the features and transform
        it (if necessary).

        :return: True if successful, False otherwise.
        """
        for fname in self._dataset_files:
            raw_data, raw_labels = self._read_data(fname)
            self._raw_data.extend(raw_data)
            self._labels.extend(raw_labels)
        self._data = self._process_features(self._raw_data)
        return True

    def get_data(self):
        """Return the transformed data and labels.

        :return: The transformed data and labels in separate lists.
        """
        return self._data, self._labels

    def get_kfold(self, num_folds, rand_seed):
        """Prepare data for processing by calculating k folds.

        :param num_folds: The number of folds to form.
        :param rand_seed: A seed for shuffling the k folds.
        :return: StratifiedKFold object representing what data set
        elements belong in each fold.
        """
        print("Calculating {0} folds...".format(num_folds))
        return StratifiedKFold(self._labels, n_folds=num_folds,
                               shuffle=True, random_state=rand_seed)

    def get_test_indices(self):
        """Return the indices in each test set fold.

        :return: Tuple of k arrays of dataset indices representing the
        testing sets.
        """
        return self._test_indices

    def get_train_indices(self):
        """Return the indices in each training set fold.

        :return: Tuple of k arrays of dataset indices representing the
        training sets.
        """
        return self._train_indices

    def get_num_norm(self):
        """Return the number of flows labeled as 'Normal'

        :return: Number as an integer.
        """
        return self._num_normal

    def get_num_attack(self):
        """Return the number of flows labeled as 'Attack'

        :return: Number as an integer.
        """
        return self._num_attack

    def _read_data(self, fname):
        """Read data from an ISCX dataset XML.

        :param fname: Name of the file to read the data from.
        :return: The data and labels.
        """
        print("Reading data from: {0}".format(fname))
        data_etree = etree.parse(fname)
        raw_data, raw_labels = self._etree_to_dict(data_etree)
        print("\tLoading complete.")
        return raw_data, raw_labels

    def _etree_to_dict(self, etree):
        """Convert an XML etree into a list of dicts.

        This method only takes care of elements, not attributes!

        :param etree: Etree object to process
        :return: Data as a list of dict.
        """
        root = etree.getroot()
        data = []
        labels = []
        for flow in root:
            flow_data = {}
            for i in range(len(flow)):
                if flow[i].tag != "Tag":
                    flow_data[flow[i].tag] = flow[i].text
                else:
                    if flow[i].text == "Normal":
                        labels.append(TagValue.Normal)
                        self._num_normal += 1
                    else:
                        labels.append(TagValue.Attack)
                        self._num_attack += 1
            data.append(flow_data)
        return data, labels

    def _process_features(self, dataset):
        """Select and process features from the ISCX data.

        :param dataset: Dataset to select features from.
        :return: The dict:list of feature sets.
        """
        print("Processing features from data...")
        feature_set = {}
        feature_set["TotalSourceBytes TotalDestinationBytes"] = \
            iscx_features.src_bytes_dst_bytes(dataset)
        feature_set["TotalSourceBytes TotalSourcePackets"] = \
            iscx_features.src_bytes_src_pckts(dataset)
        feature_set["SourceBytes-per-Packet " \
                    "DestinationBytes-per-Packet"] = \
            iscx_features.src_bpp_dst_bpp(dataset)
        feature_set["SourceBytes FlowDuration"] = \
            iscx_features.src_bytes_flow_duration(dataset)
        feature_set["log(SourceBytes) FlowDuration"] = \
            iscx_features.log_src_bytes_flow_duration(dataset)
        feature_set["SourceBytes log(FlowDuration)"] = \
            iscx_features.src_bytes_log_flow_duration(dataset)
        return feature_set


class TagValue:
    """Enum for the dataset tag labels.
    """
    Normal = 0
    Attack = 1
