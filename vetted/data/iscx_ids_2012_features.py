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
import math

__author__ = "Jarrod N. Bakker"


"""This file contains functions to produce different feature sets.
"""


def src_bytes_dst_bytes(data):
    """Return the totalSourceBytes and totalDestinationBytes as a
    feature set.

    :param data: The data set to manipulate.
    :return: List of the transformed features.
    """
    print("\tTotal Source Bytes, Total Destination Bytes")
    features = ["totalSourceBytes", "totalDestinationBytes"]
    return _return_features(data, features)


def log_src_bytes_flow_duration(data):
    """Return the log(source bytes) and flow duration as a feature set.

    :param data: The data set to manipulate.
    :return: List of the transformed features.
    """
    print("\tlog(Source Bytes), Flow Duration")
    features = ["totalSourceBytes", "startDateTime", "stopDateTime"]
    selected_data = _return_features(data, features)
    transf_data = []
    for flow in selected_data:
        new_entry = []
        src_bytes = 0
        try:
            src_bytes = math.log(float(flow[0]))
        except ValueError:
            pass
        new_entry.append(src_bytes)
        start_dt = datetime.strptime(flow[1], "%Y-%m-%dT%H:%M:%S")
        stop_dt = datetime.strptime(flow[2], "%Y-%m-%dT%H:%M:%S")
        duration = (stop_dt-start_dt).seconds
        new_entry.append(duration)
        transf_data.append(new_entry)
    return transf_data


def tsb_tsp_fl(data):
    """Return the totalSourceBytes, totalSourcePackets and flow
    duration as a feature set.

    :param data: The data set to manipulate.
    :return: List of the transformed features.
    """
    print("\ttotalSourceBytes, totalSourcePackets, Flow Duration")
    features = ["totalSourceBytes", "totalSourcePackets",
                "startDateTime", "stopDateTime"]
    selected_data = _return_features(data, features)
    transf_data = []
    for flow in selected_data:
        new_entry = flow[0:2]  # copy in the first 2 elements
        start_dt = datetime.strptime(flow[2], "%Y-%m-%dT%H:%M:%S")
        stop_dt = datetime.strptime(flow[3], "%Y-%m-%dT%H:%M:%S")
        duration = (stop_dt-start_dt).seconds
        new_entry.append(duration)
        transf_data.append(new_entry)
    return transf_data


def ltsb_tsp_fl(data):
    """Return the log(totalSourceBytes), totalSourcePackets and flow
    duration as a feature set.

    :param data: The data set to manipulate.
    :return: List of the transformed features.
    """
    print("\tlog(totalSourceBytes), totalSourcePackets, Flow Duration")
    features = ["totalSourceBytes", "totalSourcePackets",
                "startDateTime", "stopDateTime"]
    selected_data = _return_features(data, features)
    transf_data = []
    for flow in selected_data:
        new_entry = []
        src_bytes = 0
        try:
            src_bytes = math.log(float(flow[0]))
        except ValueError:
            pass
        new_entry.append(src_bytes)
        new_entry.append(flow[1])
        start_dt = datetime.strptime(flow[2], "%Y-%m-%dT%H:%M:%S")
        stop_dt = datetime.strptime(flow[3], "%Y-%m-%dT%H:%M:%S")
        duration = (stop_dt-start_dt).seconds
        new_entry.append(duration)
        transf_data.append(new_entry)
    return transf_data


def ltsb_ltsp_fl(data):
    """Return the log(totalSourceBytes), log(totalSourcePackets) and
    flow duration as a feature set.

    :param data: The data set to manipulate.
    :return: List of the transformed features.
    """
    print("\tlog(totalSourceBytes), log(totalSourcePackets), Flow "
          "Duration")
    features = ["totalSourceBytes", "totalSourcePackets",
                "startDateTime", "stopDateTime"]
    selected_data = _return_features(data, features)
    transf_data = []
    for flow in selected_data:
        new_entry = []
        src_bytes = 0
        try:
            src_bytes = math.log(float(flow[0]))
        except ValueError:
            pass
        new_entry.append(src_bytes)
        src_pckts = 0
        try:
            src_pckts = math.log(float(flow[1]))
        except ValueError:
            pass
        new_entry.append(src_pckts)
        start_dt = datetime.strptime(flow[2], "%Y-%m-%dT%H:%M:%S")
        stop_dt = datetime.strptime(flow[3], "%Y-%m-%dT%H:%M:%S")
        duration = (stop_dt-start_dt).seconds
        new_entry.append(duration)
        transf_data.append(new_entry)
    return transf_data


def tsb_tdb_fl(data):
    """Return the totalSourceBytes, totalDestinationBytes and flow
    duration as a feature set.

    :param data: The data set to manipulate.
    :return: List of the transformed features.
    """
    print("\ttotalSourceBytes, totalDestinationBytes, Flow Duration")
    features = ["totalSourceBytes", "totalDestinationBytes",
                "startDateTime", "stopDateTime"]
    selected_data = _return_features(data, features)
    transf_data = []
    for flow in selected_data:
        new_entry = flow[0:2]  # copy in the first 2 elements
        start_dt = datetime.strptime(flow[2], "%Y-%m-%dT%H:%M:%S")
        stop_dt = datetime.strptime(flow[3], "%Y-%m-%dT%H:%M:%S")
        duration = (stop_dt-start_dt).seconds
        new_entry.append(duration)
        transf_data.append(new_entry)
    return transf_data


def tsb_tsp_tdb_tdp_fl(data):
    """Return the totalSourceBytes, totalSourcePackets,
    totalDestinationBytes, totalDestinationPackets and flow duration
    as a feature set.

    :param data: The data set to manipulate.
    :return: List of the transformed features.
    """
    print("\ttotalSourceBytes, totalSourcePackets, "
          "totalDestinationBytes, totalDestinationPackets, "
          "Flow Duration")
    features = ["totalSourceBytes", "totalSourcePackets",
                "totalDestinationBytes", "totalDestinationPackets",
                "startDateTime", "stopDateTime"]
    selected_data = _return_features(data, features)
    transf_data = []
    for flow in selected_data:
        new_entry = flow[0:4]  # copy in the first 4 elements
        start_dt = datetime.strptime(flow[4], "%Y-%m-%dT%H:%M:%S")
        stop_dt = datetime.strptime(flow[5], "%Y-%m-%dT%H:%M:%S")
        duration = (stop_dt-start_dt).seconds
        new_entry.append(duration)
        transf_data.append(new_entry)
    return transf_data


def _return_features(data, features):
    """Select specific raw features from the data

    :param data: The data set to manipulate.
    :param features: A list of ISXC 2012 IDS specific features.
    :return: List of data with just the chosen features in the order
             they were requested.
    """
    processed_data = []
    for flow in data:
        new_entry = []
        for f in features:
            new_entry.append(flow[f])
        processed_data.append(new_entry)
    return processed_data
