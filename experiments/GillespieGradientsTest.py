from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
from autograd import value_and_grad
from autograd import grad

#TODO: make gradient work in parallel
#how to parallelize the gradient of the loss function?
#we could say : skip taking the gradient of gillespieGrad.run_simulation(*parameters).  Delegate it to an existing function
#inside of that function parallelize and have each of the processors compute its path's gradient, then average and return
#the parallel-computed gradient up the stack.
#TODO: install optimizers package and start using adam


def gillespieGradientWalk():

    np.set_printoptions(precision=2)
    setup = Setup(yaml_file_name="lotka_volterra.yaml")
    propensities = setup.get_propensity_list()
    parameters = setup.get_parameter_list()
    species = setup.get_species()
    incr = setup.get_increments()
    nPaths = 2 #setup.get_number_of_paths()
    T = 2.0 #setup.get_time_horizon()
    seed = 100
    numProc = 1

    my_gillespie = Gillespie(a=species[0],b=species[1],propensities=propensities,
                             increments=incr,nPaths = nPaths,T=T,useSmoothing=True, seed = seed, numProc = numProc)
    observed_data = my_gillespie.run_simulation(*parameters)

    starting_parameters = [x for x in parameters]
    idx = 0
    starting_parameters[idx] = parameters[idx]+0.2
    dw = 0.0001
    prev_loss_function_grad = 100001.0
    loss_function_grad = 100000.0
    print "{} \n".format(np.array(observed_data[:10]))
    cnt=0

    def lossFunction(*parameters):

        gillespieGrad = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr,
                                  nPaths = nPaths,T=T,useSmoothing=True, seed = seed, numProc = numProc )

        simulated_data = gillespieGrad.run_simulation(*parameters)

        return sum((np.array(simulated_data)-np.array(observed_data))**2)

    while cnt<40:

        print "prev grad {} current grad {} starting_parameters[0] {}" \
            .format(prev_loss_function_grad,loss_function_grad,starting_parameters[0])

        prev_loss_function_grad = loss_function_grad
        lossFunctionGrad = value_and_grad(lossFunction,idx)
        value, gradient = lossFunctionGrad(*starting_parameters)
        starting_parameters[idx] = starting_parameters[idx]-gradient*dw
        print "\n Loss Function Value: \n {0} \n Gradient: \n {1} ".format(value,gradient)
        cnt+=1

if __name__ == "__main__":
    gillespieGradientWalk()
