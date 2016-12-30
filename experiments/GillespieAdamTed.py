from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
from autograd.util import flatten_func
import math
from autograd import value_and_grad
#from gillespie.GillespieAdam import adam
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import pdb

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
np.set_printoptions(precision=4)

def adam(lossValueAndGrad, init_params, callback=None, num_iters=200,
         step_size=np.array([0.05,0.05,0.05]), b1=0.9, b2=0.999, eps=10 ** -8):

    flattened_value_and_grad, unflatten, x = flatten_func(lossValueAndGrad, init_params)
    #flattened_value_and_grad1, unflatten1, x1 = flatten_func(lossValueAndGrad[1], init_params)
    #flattened_value_and_grad2, unflatten2, x2 = flatten_func(lossValueAndGrad[2], init_params)

    #pdb.set_trace()
    cost_list = []
    #cost_list.append(1e10)
    param1 = []
    param2 = []
    param3 = []
    #param1.append(init_params[0])
    #param2.append(init_params[1])
    #param3.append(init_params[2])
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    update = 0
    #step_size[1:] = 0.0
    for i in range(1,num_iters):

        #if i % 10 == 0:
        #    step_size*=0.5

        #if cost_list[-1] < 5000:
        #    break

        g = flattened_value_and_grad(x, i)
        cost = g[0]
        g = g[1:]
        #g[1] = flattened_value_and_grad(x1, i)[2]
        #g[2] = flattened_value_and_grad(x2, i)[3]
        m = b1 * g + (1-b1) * m  # First  moment estimate.
        v = b2 * (g ** 2) + (1- b2) * v  # Second moment estimate.
        gamma = math.sqrt(1-(1-b2)**i)/(1-(1-b1)**i)
        print "iteration {} cost {} parameters {} log update {} ".format(i, cost, unflatten(x), update)
        #print "iteration {} cost {} parameters {} log update {} ".format(i, cost, np.exp(unflatten(x)), update)
        update = step_size*gamma*m/np.sqrt(i*v)
        #pdb.set_trace()
        if callback: callback(unflatten(x), i, unflatten(g))
        x[0] = x[0] - update[0]
        #x = x - update
        unflattened_x = unflatten(x)
        cost_list.append(cost)
        param1.append(unflattened_x[0])
        param2.append(unflattened_x[1])
        param3.append(unflattened_x[2])
        #if cost_list[-1] < 5000:
        #   break
        

    return cost_list,param1,param2,param3


def gillespieGradientWalk(n_iterations):

    np.set_printoptions(precision=4)
    setup = Setup(yaml_file_name="lotka_volterra.yaml")
    propensities = setup.get_propensity_list()
    parameters = np.array(setup.get_parameter_list())
    
    log_parameters = np.log( parameters )
    initial_log_parameters = log_parameters.copy() 
    
    log_parameters[0] += 0.5*np.random.randn(1) 
    
    print "INIT LOG PARAMS"
    print initial_log_parameters
    #print "INIT  PARAMS"
    #print np.exp(initial_log_parameters)
    
    species = setup.get_species()
    incr = setup.get_increments()
    nPaths = 1 #setup.get_number_of_paths()
    T = 1.0 #setup.get_time_horizon()
    seed = 100
    numProc = 1

    my_gillespie = Gillespie(species=species,propensities=propensities,
                             increments=incr,nPaths = nPaths,T=T,useSmoothing=True, seed = seed, numProc = numProc)

    observed_data = my_gillespie.run_simulation(parameters)
    #seed += 1
    #starting_parameters = [x for x in log_parameters]
    idx = 0
    #starting_parameters[0] = parameters[0]+parameters[0]*0.2
    #starting_parameters[1] = parameters[1]+parameters[1]*0.2
    #starting_parameters[2] = parameters[2]+parameters[2]*0.2
    
    print "{} \n".format(np.array(observed_data[:10]))
    #parameters = np.array(parameters)
    #starting_parameters = np.array(starting_parameters)

    def lossFunction(log_parameters,one_more):

        gillespieGrad = Gillespie(species=species,propensities=propensities,increments=incr,
                                  nPaths = nPaths,T=T,useSmoothing=True, seed = seed, numProc = numProc )

        simulated_data = gillespieGrad.run_simulation( np.exp(log_parameters) )

        return np.sum(np.square( (np.array(simulated_data)-np.array(observed_data))/1000.0))

    lossFunctionValueAndGrad = [value_and_grad(lossFunction,idx) for idx in range(3)]

    cost_list,param0,param1,param2 = adam(lossFunctionValueAndGrad[0],log_parameters, num_iters=n_iterations)

    fig,(axC, ax0, ax1, ax2) = plt.subplots(nrows=4,sharex=True)
    x = [x for x in range(len(param1))]

    p0 = [parameters[0] for _ in range(len(x))]
    p1 = [parameters[1] for _ in range(len(x))]
    p2 = [parameters[2] for _ in range(len(x))]

    axC.plot(x,cost_list,label="Cost",linewidth=2)
    axC.set_title("Loss")

    ax0.plot(x,param0,label="Parameter 0",linewidth=2)
    ax0.plot(x,np.log(p0),label="Actual Value ",linewidth=4)
    ax0.set_title("c1:  True = {}, Start = {}, Result = {}".format(initial_log_parameters[0],log_parameters[0],param0[-1]))

    ax1.plot(x,param1,label="Parameter 1",linewidth=2)
    ax1.plot(x,np.log(p1),label="Actual Value ",linewidth=4)
    ax1.set_title("c2: True = {}, Start = {}, Result = {}".format(initial_log_parameters[1],log_parameters[1],param1[-1]))

    ax2.plot(x,param2,label="Parameter 2",linewidth=2)
    ax2.plot(x,np.log(p2),label="Actual Value ",linewidth=4)
    ax2.set_title("c3: True = {}, Start = {}, Result = {}".format(initial_log_parameters[2],log_parameters[2],param2[-1]))

    plt.savefig("convergence.png")
    plt.show()

if __name__ == "__main__":
    gillespieGradientWalk(20)


