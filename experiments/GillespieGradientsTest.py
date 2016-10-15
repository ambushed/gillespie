from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
from autograd import value_and_grad

def gillespieGradientWalk():

    np.set_printoptions(precision=4)
    setup = Setup(yaml_file_name="lotka_volterra.yaml")
    propensities = setup.get_propensity_list()
    parameters = np.array(setup.get_parameter_list())
    species = setup.get_species()
    incr = setup.get_increments()
    nPaths = 2 #setup.get_number_of_paths()
    T = 2.0 #setup.get_time_horizon()
    seed = 100
    numProc = 1

    idx = 1

    my_gillespie = Gillespie(a=species[0],b=species[1],propensities=propensities,
                             increments=incr,nPaths = nPaths,T=T,useSmoothing=True, seed = seed, numProc = numProc)
    observed_data = my_gillespie.run_simulation(np.log(parameters))

    starting_parameters = [x for x in parameters]
    starting_parameters[idx] = parameters[idx]+parameters[idx]*0.05
    starting_parameters = np.array(starting_parameters)
    dw = np.array([0.0001,0.000000001,0.0001])
    print "{} \n".format(np.array(observed_data[:10]))
    starting_parameters = np.array(starting_parameters)

    def lossFunction(parameters):

        gillespieGrad = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr,
                                  nPaths = nPaths,T=T,useSmoothing=True, seed = seed, numProc = numProc )

        simulated_data = gillespieGrad.run_simulation(parameters)

        return sum((np.array(simulated_data)-np.array(observed_data))**2)

    i=0
    while i<40:

        print "Iteration {}. Current value of a parameter at index {} is {}".format(i, idx,starting_parameters[idx])

        lossFunctionGrad = value_and_grad(lossFunction)
        value, gradient = lossFunctionGrad(np.log(starting_parameters))
        starting_parameters[idx] = starting_parameters[idx]-gradient[idx]*dw[idx]
        print "\n Loss Function Value: \n {0} \n Gradient: \n {1} ".format(value,gradient)
        i+=1

if __name__ == "__main__":
    gillespieGradientWalk()
