from gillespie import Setup
from gillespie import Gillespie
from gillespie.GillespiePrior import GillespiePrior
from gillespie.GillespieAdam import adam
from autograd import value_and_grad
from functools import partial
import autograd.numpy as np

model_file_name = "lotka_volterra.yaml"
setup = Setup(yaml_file_name=model_file_name)

propensities = setup.get_propensity_list()
original_parameters = np.array(setup.get_parameter_list())
species = setup.get_species()
incr = setup.get_increments()
nPaths = setup.get_number_of_paths()
T = setup.get_time_horizon()
seed = 100
numProc = 1
num_adam_iters = 20

observed_data = None

def generateData():

    my_gillespie = Gillespie(species=species,propensities=propensities,
                             increments=incr,nPaths = nPaths,T=T,useSmoothing=True, seed = seed, numProc = numProc)

    observed_data = my_gillespie.run_simulation(original_parameters)
    return observed_data

def lossFunction(parameters, dummy):

    gillespieGrad = Gillespie(species=species,propensities=propensities,increments=incr,
                              nPaths = nPaths,T=T,useSmoothing=True, seed = seed, numProc = numProc )

    simulated_data = gillespieGrad.run_simulation(parameters)

    return sum(0.5*(np.array(simulated_data)-np.array(observed_data))**2)

def run_path(parameters,idx):

    global num_adam_iters
    path_parameters = parameters[idx]
    lossFunctionGrad = value_and_grad(lossFunction,idx)
    cost_list,param0,param1,param2 = adam(lossFunctionGrad, path_parameters, num_iters=num_adam_iters)
    return cost_list[-1],param0[-1],param1[-1],param2[-2]

def get_jacobians(parameters,idx):

    path_parameters = parameters[idx]
    my_gillespie = Gillespie(species=species,propensities=propensities,increments=incr, nPaths = nPaths,T=T,useSmoothing=True, seed = seed)
    gradients = my_gillespie.take_gradients(path_parameters)
    return gradients

def gillespieOMC(n_samples = 1000):

    global observed_data

    observed_data = generateData()
    parameter_count = len(setup.get_propensity_list())
    prior = GillespiePrior(n_samples=n_samples,parameter_bounds=[(1,2)]*parameter_count)
    parameter_space = prior.sample()

    runner = partial(run_path, parameter_space)
    params = map(runner,range(0,n_samples))

    zipped_params = zip(*params)

    runner_for_jacobians = partial(get_jacobians, zipped_params)
    jacobians = map(runner_for_jacobians,range(0,n_samples))

if __name__ == "__main__":

    gillespieOMC(2)

