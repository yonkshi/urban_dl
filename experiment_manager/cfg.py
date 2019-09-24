from argparse import ArgumentParser
from tabulate import tabulate
from collections import OrderedDict
import yaml
_config_data = {}

class HPConfig():
    '''
    A hyperparameter config object
    '''
    def __init__(self):
        self.data = {}
        self.argparser = ArgumentParser()

    def create_hp(self, name, value, argparse=False, argparse_args={}):
        '''
        Creates a new hyperparameter, optionally sourced from argparse external arguments
        :param name:
        :param value:
        :param argparse:
        :param argparse_args:
        :return:
        '''
        self.data[name] = value
        if argparse:
            datatype = type(value)
            # Handle boolean type
            if datatype == bool:
                self.argparser.add_argument(f'--{name}', action='store_true', *argparse_args)
            else:
                self.argparser.add_argument(f'--{name}', type=datatype, *argparse_args)

    def parse_args(self):
        '''
        Performs a parse operation from the program arguments
        :return:
        '''
        args = self.argparser.parse_known_args()[0]
        for key, value in args.__dict__.items():
            # Arg not present, using default
            if value is None: continue
            self.data[key] = value

    def __str__(self):
        '''
        Converts the HP into a human readable string format
        :return:
        '''
        table = {'hyperparameter': self.data.keys(),
                'values': list(self.data.values()),
                 }
        return tabulate(table, headers='keys', tablefmt="fancy_grid", )


    def save_yml(self, file_path):
        '''
        Save HP config to a yaml file
        :param file_path:
        :return:
        '''
        with open(file_path, 'w') as file:
            yaml.dump(self.data, file, default_flow_style=False)

    def load_yml(self, file_path):
        '''
        Load HP Config from a yaml file
        :param file_path:
        :return:
        '''
        with open(file_path, 'r') as file:
            yml_hp = yaml.safe_load(file)

        for hp_name, hp_value in yml_hp.items():
            self.data[hp_name] = hp_value

    def __getattr__(self, name):
        return self.data[name]

def config(name='default') -> HPConfig:
    '''
    Retrives a configuration (optionally, creating it) of the run. If no `name` provided, then 'default' is used
    :param name: Optional name of the
    :return: HPConfig object
    '''
    # Configuration doesn't exist yet
    if name not in _config_data.keys():
        _config_data[name] = HPConfig()
    return _config_data[name]

def load_from_yml():
    '''
    Load a HPConfig from a YML file
    :return:
    '''
    pass