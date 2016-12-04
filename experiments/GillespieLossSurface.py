from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
from autograd import value_and_grad
from gillespie.GillespieAdam import adam
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
np.set_printoptions(precision=4)
import decimal

def drange(x,y,jump):
    while x<y:
        yield x
        x+=jump

def gillespieGradientWalk():

    np.set_printoptions(precision=4)
    setup = Setup(yaml_file_name="lotka_volterra.yaml")
    propensities = setup.get_propensity_list()
    parameters = np.array(setup.get_parameter_list())
    species = setup.get_species()
    incr = setup.get_increments()
    nPaths = 1 #setup.get_number_of_paths()
    T = 1.0 #setup.get_time_horizon()
    seed = 100
    numProc = 1
    idx = 0

    my_gillespie = Gillespie(species=species,propensities=propensities,
                             increments=incr,nPaths = nPaths,T=T,useSmoothing=True, seed = seed, numProc = numProc)

    observed_data = my_gillespie.run_simulation(parameters)

    param_0_range = [x for x in drange(0.001,4.0,0.001)]
    param_1_range = [x for x in drange(0.001,4.0,0.001)]
    param_2_range = [x for x in drange(0.001,4.0,0.001)]

    def lossFunction(parameters,dummy=None):

        gillespieGrad = Gillespie(species=species,propensities=propensities,increments=incr,
                                  nPaths = nPaths,T=T,useSmoothing=True, seed = seed, numProc = numProc )

        simulated_data = gillespieGrad.run_simulation(parameters)

        #return sum(0.5*(np.array(simulated_data)-np.array(observed_data))**2)
        return np.sum(np.square( (np.array(simulated_data)-np.array(observed_data))))

    from math import log
    results = [(log(x),log(y),log(x),lossFunction([x,y,z])) for x in param_0_range for y in param_1_range for z in param_2_range]
    the_file = open("loss_func_values.txt",'w')
    for item in results:
        the_file.write("{}\n".format(item))

if __name__ == "__main__":
    gillespieGradientWalk()

