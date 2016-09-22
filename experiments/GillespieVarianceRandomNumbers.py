from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
import pylab as plt

def gillespieRandomnessVariance():

    setup = Setup(yaml_file_name="lotka_volterra.yaml")
    propensities = setup.get_propensity_list()
    parameters = np.array(setup.get_parameter_list())
    species = setup.get_species()
    incr = setup.get_increments()
    nPaths = 100 #setup.get_number_of_paths()
    T = setup.get_time_horizon()
    seed = 4000
    seed2 = 600000
    stride = 1000
    numSteps = 101

    timeGrid = np.linspace(0, T, numSteps)[1:]
    a_diff = []
    b_diff = []
    my_gillespie = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr,nPaths = nPaths,T=T,useSmoothing=False,numSteps = numSteps,numProc = 2, seed = seed)
    discrete_sim_data = my_gillespie.run_simulation(parameters)
    base_a = discrete_sim_data[:numSteps-1]
    base_b = discrete_sim_data[numSteps-1:]
    for i in range(5):
        my_gillespie = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr,nPaths = nPaths,T=T,useSmoothing=False,numSteps = numSteps, numProc = 2, seed = seed2+i*stride)
        discrete_sim_data = my_gillespie.run_simulation(parameters)
        discrete_a = discrete_sim_data[:numSteps-1]
        discrete_b = discrete_sim_data[numSteps-1:]
        a_diff.append([a - b for a, b in zip(discrete_a, base_a)])
        b_diff.append([a - b for a, b in zip(discrete_b, base_b)])
        if i % 5 == 0:
            print("!")

    a_matrix = np.array(a_diff)
    b_matrix = np.array(b_diff)

    a_mean = a_matrix.mean(axis=0)
    a_std = a_matrix.std(axis=0)*2
    b_mean = b_matrix.mean(axis=0)
    b_std = b_matrix.std(axis=0)*2

    fig,(ax0,ax1,ax2) = plt.subplots(nrows=3,sharex=True)
    ax0.plot(timeGrid,base_a)
    ax0.plot(timeGrid,base_b)
    ax1.set_title("A and B species")
    ax1.errorbar(timeGrid, a_mean, yerr=a_std)
    ax1.set_title("A species random numbers 2*std ")
    ax2.errorbar(timeGrid, b_mean, yerr=b_std)
    ax2.set_title("B species random numbers 2*std")

    plt.show()

if __name__ == "__main__":
    gillespieRandomnessVariance()