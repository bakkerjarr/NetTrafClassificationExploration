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

import yaml

__author__ = "Jarrod N. Bakker"

class ConfigLoader:
    """Handles the loading of configuration data from YAML files.
    """

    def __init__(self, classifier_conf):
        """Initialise.

        :param classifier_conf: Path to the classifier config file.
        """
        self._classifier_conf = classifier_conf

    def read_config(self):
        """Parse configuration file/s.

        :return: True if successful, False otherwise.
        """
        pass

    def get_classifier_config(self):
        """Return the configuration information for classifiers.

        :return: Dict of configuration information.
        """
        pass
