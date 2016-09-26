from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
from autograd import value_and_grad
from autograd import grad
from autograd.util import flatten_func

def adam(grad, init_params, callback=None, num_iters=200,
         step_size=np.array([0.01,0.000001,0.01]), b1=0.9, b2=0.999, eps=10 ** -8):

    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    flattened_grad, unflatten, x = flatten_func(grad, init_params)

    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = flattened_grad(x, i)
        if callback: callback(unflatten(x), i, unflatten(g))
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g ** 2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
        vhat = v / (1 - b2 ** (i + 1))
        update = np.diag(step_size) * mhat / (np.sqrt(vhat) + eps)

        print np.diag(update)

        x = x - np.dot(update,np.array([1,1,1]))
        print "iteration {} parameters {}".format(i,unflatten(x))

    return unflatten(x)

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
    starting_parameters[idx] = parameters[idx]+0.2
    print "{} \n".format(np.array(observed_data[:10]))

    parameters = np.array(parameters)
    starting_parameters = np.array(starting_parameters)

    def lossFunction(parameters,one_more):

        gillespieGrad = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr,
                                  nPaths = nPaths,T=T,useSmoothing=True, seed = seed, numProc = numProc )

        simulated_data = gillespieGrad.run_simulation(parameters)

        return sum((np.array(simulated_data)-np.array(observed_data))**2)

    lossFunctionGrad = grad(lossFunction,idx)
    print adam(lossFunctionGrad,starting_parameters)

if __name__ == "__main__":
    gillespieGradientWalk()