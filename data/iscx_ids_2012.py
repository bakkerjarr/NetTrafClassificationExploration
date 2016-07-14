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

from datetime import datetime
from lxml import etree
from sklearn.cross_validation import StratifiedKFold
import math
import random

__author__ = "Jarrod N. Bakker"


class ISCX2012IDS:

    _BASE_PATH = "/vol/nerg-solar/bakkerjarr/Datasets/ISCXIDS2012" \
                 "/labeled_flows_xml/"

    def __init__(self, fnames):
        """Initialise.

        :param fnames: List of dataset file names.
        """
        self._rand = random
        #self._features = ["totalSourceBytes", "totalDestinationBytes",
        #                  "totalSourcePackets",
        #                  "totalDestinationPackets"]
        #self._features = ["totalSourceBytes", "totalSourcePackets"]
        self._features = ["totalSourceBytes", "startDateTime",
                          "stopDateTime"]
        self._transform = True
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
        feat_data = self._select_features(self._raw_data, self._features)
        if self._transform:
            #trans_data = self._transform_data_bpp(feat_data)
            trans_data = self._calculate_flow_duration(feat_data)
        else:
            trans_data = feat_data
        self._data = trans_data
        return True

    def prepare_data(self, num_folds, rand_seed):
        """Prepare data for processing by calculating k folds.

        :param num_folds: The number of folds to form.
        :param rand_seed: A seed for shuffling the k folds.
        :return: True if successful, False otherwise.
        """
        print("Calculating {0} folds...".format(num_folds))
        self._skf = StratifiedKFold(self._labels, n_folds=num_folds,
                                    shuffle=True, random_state=rand_seed)
        return True

    def get_data(self):
        """Return the transformed data and labels.

        :return: The transformed data and labels in separate lists.
        """
        return self._data, self._labels

    def get_kfold(self):
        """Return the stratified k-fold dataset.

        :return: A StratifiedKFold object to use for iterating with.
        """
        return self._skf

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
        print("Loading complete.")
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

    def _select_features(self, dataset, features):
        """Select features from the ISCX data.

        :param dataset: Dataset to select features from.
        :param features: A list of features encoded using the
        FlowElement enum.
        :return: The dataset with the selected features.
        """
        print("Selecting features from data...")
        processed_dataset = []
        for flow in dataset:
            new_entry = []
            for f in features:
                new_entry.append(flow[f])
            processed_dataset.append(new_entry)
        return processed_dataset

    def _transform_data_bpp(self, data):
        """Transform the features in data flows to get the bytes per
        packets ratio.

        :param data: The dataset to transform.
        :return: Transformed data.
        """
        print("Transforming data...")
        trans_data = []
        for flow in data:
            new_entry = []
            src_bp_ratio = 0
            try:
                src_bp_ratio = float(flow[0])/float(flow[2])
            except ZeroDivisionError:
                pass
            new_entry.append(src_bp_ratio)
            dst_bp_ratio = 0
            try:
                dst_bp_ratio = float(flow[1])/float(flow[3])
            except ZeroDivisionError:
                pass
            new_entry.append(dst_bp_ratio)
            trans_data.append(new_entry)
        return trans_data

    def _calculate_flow_duration(self, data):
        """The data to calculate the flow duration time for.

        :param data: The data.
        :return: Transformed data.
        """
        print("Transforming data...")
        trans_data = []
        for flow in data:
            new_entry = []
            src_bytes = 0
            try:
                # Remember to include log(source bytes)
                src_bytes = math.log(float(flow[0]))
            except ValueError:
                pass
            new_entry.append(src_bytes)
            start_dt = datetime.strptime(flow[1], "%Y-%m-%dT%H:%M:%S")
            stop_dt = datetime.strptime(flow[2], "%Y-%m-%dT%H:%M:%S")
            duration = (stop_dt-start_dt).seconds
            new_entry.append(duration)
            trans_data.append(new_entry)
        return trans_data


class TagValue:
    """Enum for the dataset tag labels.
    """
    Normal = 0
    Attack = 1
