from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
import pylab as plt

#compare the same number of runs
#parameters, jacobian, value of the simulation (epsilon, weight)

def gillespieDiscreteVsSmoothVariance():

    setup = Setup(yaml_file_name="lotka_volterra.yaml")
    propensities = setup.get_propensity_list()
    parameters = setup.get_parameter_list()
    species = setup.get_species()
    incr = setup.get_increments()
    nPaths = 100#setup.get_number_of_paths()
    T = setup.get_time_horizon()
    idx = 0
    seed = 3000
    seed2 = 700000
    stride = 1000
    numSteps = 101

    my_gillespie = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr,
                             nPaths = nPaths, T=T, useSmoothing=False,numSteps = numSteps, seed = seed)
    discrete_sim_data = my_gillespie.run_simulation(*parameters)
    discrete_a = discrete_sim_data[:numSteps-1]
    discrete_b = discrete_sim_data[numSteps-1:]

    timeGrid = np.linspace(0, T, numSteps)[1:]
    a_diff = []
    b_diff = []
    result_a = []
    result_b = []
    for i in range(50):
        my_gillespie = Gillespie(a=species[0], b=species[1], propensities=propensities, increments=incr,
                                 nPaths=nPaths, T=T, useSmoothing=True,numSteps=numSteps, seed = seed2+i*stride)
        res = my_gillespie.run_simulation(*parameters)
        smooth_a = res[:numSteps-1]
        smooth_b = res[numSteps-1:]
        a_diff.append([a - b for a, b in zip(smooth_a, discrete_a)])
        b_diff.append([a - b for a, b in zip(smooth_b, discrete_b)])
        if i % 5 == 0:
            print("!")

    a_matrix = np.array(a_diff)
    b_matrix = np.array(b_diff)

    a_mean = a_matrix.mean(axis=0)
    a_std = a_matrix.std(axis=0)*2
    b_mean = b_matrix.mean(axis=0)
    b_std = b_matrix.std(axis=0)*2

    fig,(ax0, ax1,ax2) = plt.subplots(nrows=3,sharex=True)
    ax0.step(timeGrid, discrete_a,'green')
    ax0.plot(timeGrid, smooth_a,'green')
    ax0.step(timeGrid, discrete_b,'red')
    ax0.plot(timeGrid, smooth_b,'red')
    ax0.set_title("Original Discrete vs Smooth Simulation T: {} nPaths: {}".format(T,nPaths))
    ax1.errorbar(timeGrid, a_mean, yerr=a_std)
    ax1.set_title("Smooth A species error mean and 2*std")
    ax2.errorbar(timeGrid, b_mean, yerr=b_std)
    ax2.set_title("Smooth B species error mean and 2*std")

    plt.savefig("100_paths.png")

if __name__ == "__main__":
    gillespieDiscreteVsSmoothVariance()
