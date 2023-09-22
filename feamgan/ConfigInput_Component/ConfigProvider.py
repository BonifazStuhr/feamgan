import json

class ConfigProvider():
    """
    The ConfigProvider provides the interface for getting the various config files via a path specification.
    """

    def get_config(self, path_to_config):
        """
        Parses the config from a file with the path: path_to_config.
        :param path_to_config: (String) The path to the config file.
        :return: config: (Dictionary) The parsed config.
        """
        if path_to_config.endswith('.json'):
            return self.parse_json_config(path_to_config)
        else:
            return None

    def parse_json_config(self, path_to_json_config):
        """
        Parses the config from a json-file with the path: path_to_json_config.
        :param path_to_json_config: (String) The path to the json config file.
        :return: json_config: (Dictionary) The parsed json config.
        """
        json_file = open(path_to_json_config)
        json_config = json.load(json_file)
        json_file.close()
        return json_config