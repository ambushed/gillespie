import unittest
from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
from autograd import jacobian

#TODO: pass two seeds to the simulation

#1.  Run simulation to get "observed data" given the seeked parameters
#2.  Bump one (then more) parameter value to take it away from true value
#3.  Calculate the dG/dw and value of the Gillespie sim at the new w value
#4.  Plug into the Loss function
#5.  Do Gradient descent using the derivative of a loss function

def gillespieGradientWalk():

    setup = Setup(yaml_file_name="lotka_volterra.yaml")
    propensities = setup.get_propensity_list()
    parameters = setup.get_parameter_list()
    species = setup.get_species()
    incr = setup.get_increments()
    nPaths = 1 #setup.get_number_of_paths()
    T = 2.0 #setup.get_time_horizon()
    seed = 100
    seed2 = 100000

    tau = np.linspace(0,T,51)
    idx = 0

    my_gillespie = Gillespie(a=species[0],b=species[1],propensities=propensities,
                             increments=incr,nPaths = nPaths,T=T,useSmoothing=True, seed = seed)
    observed_data = my_gillespie.run_simulation(*parameters)

    starting_parameters = [x for x in parameters]
    starting_parameters[idx] = parameters[idx]+0.2
    dw = 0.00001
    prev_loss_function_grad = 100001.0
    loss_function_grad = 100000.0
    print observed_data[:10],"\n"
    cnt=0

    while cnt<20:

        print "prev grad {} current grad {} starting_parameters[0] {}" \
            .format(prev_loss_function_grad,loss_function_grad,starting_parameters[0])
        prev_loss_function_grad = loss_function_grad

        gillespieGrad = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr,
                                  nPaths = nPaths,T=T,useSmoothing=True, seed = seed)
        gradient = gillespieGrad.take_gradients(*starting_parameters)

        gillespieSim = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr,
                                 nPaths = nPaths,T=T,useSmoothing=True, seed = seed)
        simulated_data = gillespieSim.run_simulation(*starting_parameters)

        print "\n Simulated Data: \n {} \n Gradients: \n {} ".format(simulated_data[:10],gradient[:10])
        element_wise_grad = [(observed_data[i]-simulated_data[i])*gradient[i] for i in range(len(gradient))]

        loss_function_grad = 2*sum(element_wise_grad)

        starting_parameters[idx] = starting_parameters[idx]-loss_function_grad*dw

        cnt+=1


if __name__ == "__main__":
    gillespieGradientWalk()
