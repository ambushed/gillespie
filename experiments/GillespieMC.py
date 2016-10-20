from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
from autograd import value_and_grad
import matplotlib
matplotlib.use('Qt4Agg')
import pymc3 as pm
import theano.tensor as tt

def build_model():

    setup = Setup(yaml_file_name="lotka_volterra.yaml")
    propensities = setup.get_propensity_list()
    parameters = np.array(setup.get_parameter_list())
    species = setup.get_species()
    incr = setup.get_increments()
    nPaths = 1 #setup.get_number_of_paths()
    T = 2.0 #setup.get_time_horizon()
    seed = 100
    numProc = 1

    my_gillespie = Gillespie(species=species,propensities=propensities,
                             increments=incr,nPaths = nPaths,T=T,useSmoothing=True, seed = seed, numProc = numProc)

    observed_data = my_gillespie.run_simulation(parameters)

    with pm.Model() as model:

        c1 = pm.Uniform('c1',1.0, 2.5)
        c2 = pm.Uniform('c2',0.00001, 0.1)
        c3 = pm.Uniform('c3',0.5, 1.5)

        def lossFunction(c1,c2,c3):

            gillespieGrad = Gillespie(species=species,propensities=propensities,increments=incr,
                                      nPaths = nPaths,T=T,useSmoothing=True, seed = seed, numProc = numProc )

            parameters = [c1,c2,c3]
            simulated_data = gillespieGrad.run_simulation(parameters)

            return tt.sum(0.5*(np.array(simulated_data)-np.array(observed_data))**2)

        pm.DensityDist('x', lossFunction, observed={'c1': c1, 'c2': c2, 'c3': c3})

    return model

def gillespieMC(n_samples = 1000):

    model = build_model()
    start = model.test_point
    #h = pm.find_hessian(start, model=model)
    step = pm.Metropolis(model.vars, h, blocked=True, model=model)
    trace = pm.sample(n_samples, step, start, model=model)
    return trace

    #lossFunctionGrad = value_and_grad(lossFunction,idx)

if __name__ == "__main__":
    gillespieMC(1000)


