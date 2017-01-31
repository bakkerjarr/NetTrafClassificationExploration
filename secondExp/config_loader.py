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

    def __init__(self, config_file_path):
        """Initialise.

        :param config_file_path: Path to the classifier config file.
        """
        self._file_path = config_file_path
        self._classifier_conf = None

    def read_config(self):
        """Parse configuration file/s.

        :return: True if successful, False otherwise.
        """
        try:
            conf_file = open(self._file_path, "r")
            print("Reading configuration from: {0}".format(
                self._file_path))
            self._classifier_conf = yaml.load(conf_file)
            for cls in self._classifier_conf:
                for items in self._classifier_conf[cls]:
                    if self._classifier_conf[cls][items] == "None":
                        self._classifier_conf[cls][items] = None
        except IOError as err:
            print("ERROR: {0}".format(err))
            return False
        finally:
            conf_file.close()
        return True

    def get_classifier_config(self):
        """Return the configuration information for classifiers.

        :return: Dict of configuration information.
        """
        return self._classifier_conf
