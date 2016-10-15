from gillespie import Gillespie
from gillespie import Setup
import autograd.numpy as np
import pylab as plt

def gillespieBumpAndRevalVsAutograd():

    setup = Setup(yaml_file_name="lotka_volterra.yaml")
    propensities = setup.get_propensity_list()
    parameters = np.array(setup.get_parameter_list())
    species = setup.get_species()
    incr = setup.get_increments()
    nPaths = 1
    T = 2.0
    seed = 5000
    tau = np.linspace(0,T,51)

    delta = 0.000001
    idx = 0

    my_gillespie = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr, nPaths = nPaths,T=T,useSmoothing=True, seed = seed)
    gradients = my_gillespie.take_gradients(*np.log(parameters))
    aGradient = gradients[:50]
    bGradient = gradients[50:]

    parameters1 = [x for x in parameters]
    parameters1[idx] = parameters[idx]+delta
    my_gillespieUp = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr, nPaths = nPaths,T=T,useSmoothing=True, seed = seed)
    bumpUp = my_gillespieUp.run_simulation(np.log(parameters1))
    parameters1[idx] = parameters[idx]-delta
    my_gillespieDown = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr, nPaths = nPaths,T=T,useSmoothing=True, seed = seed)
    bumpDown = my_gillespieDown.run_simulation(np.log(parameters1))
    aBumpUp = bumpUp[:50]
    aBumpDown = bumpDown[:50]
    bBumpUp = bumpUp[50:]
    bBumpDown = bumpDown[50:]
    diffA = [(up-down)/(2*delta) for up,down in zip(aBumpUp,aBumpDown)]
    diffB = [(up-down)/(2*delta) for up,down in zip(bBumpUp,bBumpDown)]

    fig,(ax0, ax1) = plt.subplots(nrows=2,sharex=True)
    ax0.plot(tau[1:],diffA,label="BumpAndReval ",linewidth=10)
    ax0.plot(tau[1:],aGradient,label="Autograd ",linewidth=5)
    ax0.set_title("Species A. AutoGrad vs BumpAndReval T: {} nPaths: {}".format(T,nPaths))

    ax1.plot(tau[1:],diffB,label="BumpAndReval",linewidth=10)
    ax1.plot(tau[1:],bGradient,label="Autograd ",linewidth=5)
    ax1.set_title("Species B. AutoGrad vs BumpAndReval T: {} nPaths: {}".format(T,nPaths))
    plt.show()

if __name__ == "__main__":
    gillespieBumpAndRevalVsAutograd()