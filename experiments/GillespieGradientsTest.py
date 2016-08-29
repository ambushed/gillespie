from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np

#TODO: pass two seeds to the simulation

def gillespieGradientWalk():

    np.set_printoptions(precision=2)
    setup = Setup(yaml_file_name="lotka_volterra.yaml")
    propensities = setup.get_propensity_list()
    parameters = setup.get_parameter_list()
    species = setup.get_species()
    incr = setup.get_increments()
    nPaths = 1 #setup.get_number_of_paths()
    T = 2.0 #setup.get_time_horizon()
    seed = 100

    my_gillespie = Gillespie(a=species[0],b=species[1],propensities=propensities,
                             increments=incr,nPaths = nPaths,T=T,useSmoothing=True, seed = seed)
    observed_data = my_gillespie.run_simulation(*parameters)

    starting_parameters = [x for x in parameters]
    idx = 0
    starting_parameters[idx] = parameters[idx]+0.2
    dw = 0.0001
    prev_loss_function_grad = 100001.0
    loss_function_grad = 100000.0
    print "{} \n".format(np.array(observed_data[:10]))
    cnt=0

    while cnt<40:


        print "prev grad {} current grad {} starting_parameters[0] {}" \
            .format(prev_loss_function_grad,loss_function_grad,starting_parameters[0])

        if np.sign(prev_loss_function_grad)+np.sign(loss_function_grad) == 0:
            dw = dw/5.;

        prev_loss_function_grad = loss_function_grad

        gillespieGrad = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr,
                                  nPaths = nPaths,T=T,useSmoothing=True, seed = seed)
        gradient = gillespieGrad.take_gradients(*starting_parameters)

        gillespieSim = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr,
                                 nPaths = nPaths,T=T,useSmoothing=True, seed = seed)
        simulated_data = gillespieSim.run_simulation(*starting_parameters)

        print "\n Simulated Data: \n {0} \n Gradients: \n {1} ".format(np.array(simulated_data[:10]),np.array(gradient[:10]))
        element_wise_grad = [(simulated_data[i]-observed_data[i])*gradient[i] for i in range(len(gradient))]

        loss_function_grad = 2*sum(element_wise_grad)

        starting_parameters[idx] = starting_parameters[idx]-loss_function_grad*dw

        cnt+=1

if __name__ == "__main__":
    gillespieGradientWalk()
