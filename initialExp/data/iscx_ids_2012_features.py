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
    """Return the totalSourceBytes and totalDestinationBytes.

    :param data: The data set to manipulate.
    :return: List of the transformed features.
    """
    print("\tTotal Source Bytes, Total Destination Bytes")
    features = ["totalSourceBytes", "totalDestinationBytes"]
    return _return_features(data, features)


def src_bytes_src_pckts(data):
    """Return the totalSourceBytes and totalSourcePackets.

    :param data: The data set to manipulate.
    :return: List of the transformed features.
    """
    print("\tTotal Source Bytes, Total Source Packets")
    features = ["totalSourceBytes", "totalSourcePackets"]
    return _return_features(data, features)


def src_bpp_dst_bpp(data):
    """Return the source bytes per packet and destination bytes per
    packet.

    :param data: The data set to manipulate.
    :return: List of the transformed features.
    """
    print("\tSource Bytes per Packet, Destination Bytes per Packet")
    features = ["totalSourceBytes", "totalSourcePackets",
                "totalDestinationBytes", "totalDestinationPackets"]
    selected_data = _return_features(data, features)
    transf_data = []
    for flow in selected_data:
        new_entry = []
        src_bp_ratio = 0
        try:
            src_bp_ratio = float(flow[0])/float(flow[1])
        except ZeroDivisionError:
            pass
        new_entry.append(src_bp_ratio)
        dst_bp_ratio = 0
        try:
            dst_bp_ratio = float(flow[2])/float(flow[3])
        except ZeroDivisionError:
            pass
        new_entry.append(dst_bp_ratio)
        transf_data.append(new_entry)
    return transf_data


def src_bytes_flow_duration(data):
    """Return the source bytes and flow duration.

    :param data: The data set to manipulate.
    :return: List of the transformed features.
    """
    print("\tSource Bytes, Flow Duration")
    features = ["totalSourceBytes", "startDateTime", "stopDateTime"]
    selected_data = _return_features(data, features)
    transf_data = []
    for flow in selected_data:
        new_entry = []
        src_bytes = float(flow[0])
        new_entry.append(src_bytes)
        start_dt = datetime.strptime(flow[1], "%Y-%m-%dT%H:%M:%S")
        stop_dt = datetime.strptime(flow[2], "%Y-%m-%dT%H:%M:%S")
        duration = (stop_dt-start_dt).seconds
        new_entry.append(duration)
        transf_data.append(new_entry)
    return transf_data


def log_src_bytes_flow_duration(data):
    """Return the log(source bytes) and flow duration.

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


def src_bytes_log_flow_duration(data):
    """Return the log(source bytes) and flow duration.

    :param data: The data set to manipulate.
    :return: List of the transformed features.
    """
    print("\tSource Bytes, log(Flow Duration)")
    features = ["totalSourceBytes", "startDateTime", "stopDateTime"]
    selected_data = _return_features(data, features)
    transf_data = []
    for flow in selected_data:
        new_entry = []
        src_bytes = float(flow[0])
        new_entry.append(src_bytes)
        start_dt = datetime.strptime(flow[1], "%Y-%m-%dT%H:%M:%S")
        stop_dt = datetime.strptime(flow[2], "%Y-%m-%dT%H:%M:%S")
        duration = 0
        try:
            duration = math.log((stop_dt-start_dt).seconds)
        except ValueError:
            pass
        new_entry.append(duration)
        transf_data.append(new_entry)
    return transf_data


def _return_features(data, features):
    """Select specific raw features from the data

    :param data: The data set to manipulate.
    :param features: A list of ISXC 2012 IDS specific features.
    :return: List of data with just the chosen features.
    """
    processed_data = []
    for flow in data:
        new_entry = []
        for f in features:
            new_entry.append(flow[f])
        processed_data.append(new_entry)
    return processed_data
