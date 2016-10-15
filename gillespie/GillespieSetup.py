import yaml
import os
import math
import autograd.numpy as np

class Setup(object):

    def __init__(self, yaml_file_name):

        model_file = os.path.join(os.path.dirname(__file__), '../models/'+yaml_file_name)
        with open(model_file, 'r') as stream:
            try:
                self.setup = yaml.load(stream)

            except yaml.YAMLError as exc:
                print(exc)

    #this needs to return an executable object which has an attribute: parameter
    #pass parameters separately and zip them!
    def get_propensity_list(self):
        propensities = self.setup['propensities']
        return [eval(compile(prop,"",'eval')) for prop in propensities]

    def get_parameter_list(self):
        parameters = self.setup['parameters']
        return [float(p) for p in parameters]

    def get_species(self):
        species = self.setup['species']
        return [int(s) for s in species]

    def get_increments(self):
        increments = self.setup['increments']
        return [[float(i) for i in row.split(',')] for row in increments]

    def get_time_horizon(self):
        return float(self.setup['time_horizon'])

    def get_number_of_paths(self):
        return int(self.setup['number_of_paths'])
