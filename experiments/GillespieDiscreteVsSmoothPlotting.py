from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
import pylab as plt

def gillespieDiscreteVsSmoothPlot():
    setup = Setup(yaml_file_name="../models/lotka_volterra.yaml")
    propensities = setup.get_propensity_list()
    parameters = setup.get_parameter_list()
    species = setup.get_species()
    incr = setup.get_increments()
    nPaths = 2
    T = 2
    seed = 1000

    my_gillespieUp = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr, nPaths = nPaths,T=T,useSmoothing = False, seed = seed)
    aMean = my_gillespieUp.run_simulation(*parameters)
    my_gillespieSmooth = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr, nPaths = nPaths,T=T,useSmoothing = True, seed = seed)
    aMeanS = my_gillespieSmooth.run_simulation(*parameters)

    tau = np.linspace(0,T,51)

    plt.plot(tau[1:],aMeanS,label="Smooth A")
    plt.step(tau[1:],aMean,label='Step A')
    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()

if __name__ == "__main__":
    gillespieDiscreteVsSmoothPlot()