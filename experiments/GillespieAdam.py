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

    true_log_parameters = np.log(parameters)
    log_parameters = np.log( parameters )

    log_parameters[0] += 0.5*np.random.randn(1)
    log_parameters[1] += 0.5*np.random.randn(1)
    log_parameters[2] += 0.5*np.random.randn(1)

    print "True Log-Params: {}".format(true_log_parameters)
    print "Where we start: {} \n".format(log_parameters)

    def lossFunction(log_parameters,dummy):

        gillespieGrad = Gillespie(species=species,propensities=propensities,increments=incr,
                                  nPaths = nPaths,T=T,useSmoothing=True, seed = seed, numProc = numProc )

        simulated_data = gillespieGrad.run_simulation(np.exp(log_parameters))

        #return sum(0.5*(np.array(simulated_data)-np.array(observed_data))**2)
        return np.sum(np.square( (np.array(simulated_data)-np.array(observed_data))/100.0))

    lossFunctionGrad = value_and_grad(lossFunction,idx)

    cost_list,param0,param1,param2 = adam(lossFunctionGrad,np.array(log_parameters).copy(), num_iters=40)

    fig,(axC, ax0, ax1, ax2) = plt.subplots(nrows=4,sharex=True)
    x = [x for x in range(len(param1))]

    p0 = [true_log_parameters[0] for _ in range(len(x))]
    p1 = [true_log_parameters[1] for _ in range(len(x))]
    p2 = [true_log_parameters[2] for _ in range(len(x))]

    axC.plot(x,cost_list,label="Cost",linewidth=2)
    axC.set_title("Loss")

    ax0.plot(x,param0,label="Parameter 0",linewidth=2)
    ax0.plot(x,p0,label="Actual Value ",linewidth=4)
    ax0.set_title("c1:  True = {}, Start = {}, Result = {}".format(true_log_parameters[0],log_parameters[0],param0[-1]))

    ax1.plot(x,param1,label="Parameter 1",linewidth=2)
    ax1.plot(x,p1,label="Actual Value ",linewidth=4)
    ax1.set_title("c2: True = {}, Start = {}, Result = {}".format(true_log_parameters[1],log_parameters[1],param1[-1]))

    ax2.plot(x,param2,label="Parameter 2",linewidth=2)
    ax2.plot(x,p2,label="Actual Value ",linewidth=4)
    ax2.set_title("c3: True = {}, Start = {}, Result = {}".format(true_log_parameters[2],log_parameters[2],param2[-1]))

    plt.savefig("convergence.png")
    plt.show()

if __name__ == "__main__":
    gillespieGradientWalk()


