from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
from autograd import value_and_grad
from autograd.util import flatten_func
import matplotlib
import math
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def adam(grad, init_params, callback=None, num_iters=200,
         step_size=np.array([0.5]*3), b1=0.9, b2=0.999, eps=10 ** -8):

    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    flattened_grad, unflatten, x = flatten_func(grad, init_params)

    cost_list = []
    param1 = []
    param2 = []
    param3 = []
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    cost = 1000000000
    for i in range(1,num_iters):
        print "iteration {} cost {} parameters {}".format(i, cost, unflatten(np.exp(x)))
        g = flattened_grad(x, i)
        cost = g[0]
        g = g[1:]
        m = b1 * g + (1-b1) * m  # First  moment estimate.
        v = b2 * (g ** 2) + (1- b2) * v  # Second moment estimate.
        gamma = math.sqrt(1-(1-b2)**i)/(1-(1-b1)**i)
        update = step_size*gamma*m/np.sqrt(i*v)
        if callback: callback(unflatten(x), i, unflatten(g))
        x = x - update
        unflattened_x = unflatten(x)
        cost_list.append(cost)
        param1.append(np.exp(unflattened_x[0]))
        param2.append(np.exp(unflattened_x[1]))
        param3.append(np.exp(unflattened_x[2]))


    return cost_list,param1,param2,param3

def gillespieGradientWalk():

    np.set_printoptions(precision=4)
    setup = Setup(yaml_file_name="lotka_volterra.yaml")
    propensities = setup.get_propensity_list()
    parameters = np.array(setup.get_parameter_list())
    species = setup.get_species()
    incr = setup.get_increments()
    nPaths = 1 #setup.get_number_of_paths()
    T = 2.0 #setup.get_time_horizon()
    seed = 100
    numProc = 1

    my_gillespie = Gillespie(a=species[0],b=species[1],propensities=propensities,
                             increments=incr,nPaths = nPaths,T=T,useSmoothing=True, seed = seed, numProc = numProc)

    observed_data = my_gillespie.run_simulation(parameters)

    starting_parameters = [x for x in parameters]
    idx = 0
    starting_parameters[0] = parameters[0]+parameters[0]*0.2
    starting_parameters[1] = parameters[1]+parameters[1]*0.2
    starting_parameters[2] = parameters[2]+parameters[2]*0.2
    print "{} \n".format(np.array(observed_data[:10]))

    parameters = np.array(parameters)
    starting_parameters = np.array(starting_parameters)

    def lossFunction(parameters,one_more):

        gillespieGrad = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr,
                                  nPaths = nPaths,T=T,useSmoothing=True, seed = seed, numProc = numProc )

        simulated_data = gillespieGrad.run_simulation(parameters)

        return sum(0.5*(np.array(simulated_data)-np.array(observed_data))**2)

    lossFunctionGrad = value_and_grad(lossFunction,idx)

    cost_list,param0,param1,param2 = adam(lossFunctionGrad,starting_parameters, num_iters=50)
    fig,(axC, ax0, ax1, ax2) = plt.subplots(nrows=4,sharex=True)
    x = [x for x in range(len(param1))]

    p0 = [parameters[0] for _ in range(len(x))]
    p1 = [parameters[1] for _ in range(len(x))]
    p2 = [parameters[2] for _ in range(len(x))]

    axC.plot(x,cost_list,label="Cost",linewidth=2)
    axC.set_title("Loss")

    ax0.plot(x,param0,label="Parameter 0",linewidth=2)
    ax0.plot(x,p0,label="Actual Value ",linewidth=4)
    ax0.set_title("Parameter 0 = {}".format(parameters[0]))

    ax1.plot(x,param1,label="Parameter 1",linewidth=2)
    ax1.plot(x,p1,label="Actual Value ",linewidth=4)

    ax2.plot(x,param2,label="Parameter 2",linewidth=2)
    ax2.plot(x,p2,label="Actual Value ",linewidth=4)

    plt.savefig("convergence.png")
    plt.show()

if __name__ == "__main__":
    gillespieGradientWalk()


