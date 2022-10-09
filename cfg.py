import yaml


class Cfg:
    def __init__(self):
        """
        Load config file by the path

        :param path: the path of config.yaml
        """
        with open('./config.yaml', 'r') as yml:
            config = yaml.safe_load(yml)

        self.cfg = config
