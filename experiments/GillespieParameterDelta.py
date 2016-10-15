from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
import pylab as plt

def gillespieParameterVariance():

    setup = Setup(yaml_file_name="lotka_volterra.yaml")
    propensities = setup.get_propensity_list()
    parameters = np.array(setup.get_parameter_list())
    species = setup.get_species()
    incr = setup.get_increments()
    nPaths = 2#setup.get_number_of_paths()
    T = 2.0#setup.get_time_horizon()
    seed = 50
    seed2 = 5000

    tau = np.linspace(0,T,51)
    idx = 0

    my_gillespie = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr,nPaths = nPaths,T=T,useSmoothing=False, seed = seed)
    discrete_sim_data = my_gillespie.run_simulation(np.log(parameters))
    discrete_a = discrete_sim_data[:50]
    discrete_b = discrete_sim_data[50:]

    a_data = []
    b_data = []
    offset = -0.5
    step = 0.05
    for i in range(20):
        starting_parameters = [x for x in parameters]
        starting_parameters[idx] = parameters[idx]+step*i+offset
        starting_parameters = np.array(starting_parameters)
        my_gillespie = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr,nPaths = nPaths,T=T, useSmoothing=True, seed = seed2 + i*10000)
        res = my_gillespie.run_simulation(np.log(starting_parameters))
        a_data.append(res[:50])
        b_data.append(res[50:])
        print(starting_parameters[idx])

    timeGrid = np.linspace(0,T,51)[1:]

    for i in range(len(a_data)):
        plt.plot(timeGrid, a_data[i], 'green', label = "smooth {}".format(i))

    plt.plot(timeGrid, discrete_a, 'red', label = "discrete", marker='o')

    plt.show()

if __name__ == "__main__":
    gillespieParameterVariance()

