import autograd.numpy as np
from autograd.util import flatten_func
import math

"""Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
It's basically RMSprop with momentum and some correction terms."""

def adam(grad, init_params, callback=None, num_iters=200,
         step_size=np.array([0.5,0.5,0.5]), b1=0.9, b2=0.999, eps=10 ** -8):

    flattened_grad, unflatten, x = flatten_func(grad, init_params)

    cost_list = []
    cost_list.append(1e10)
    param1 = []
    param2 = []
    param3 = []
    param1.append(init_params[0])
    param2.append(init_params[1])
    param3.append(init_params[2])
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    update = 0
    #step_size[1:] = 0.0
    for i in range(1,num_iters):

        #if i % 10 == 0:
        #    step_size*=0.5

        if cost_list[-1] < 5000:
            break

        g = flattened_grad(x, i)
        cost = g[0]
        g = g[1:]
        m = b1 * g + (1-b1) * m  # First  moment estimate.
        v = b2 * (g ** 2) + (1- b2) * v  # Second moment estimate.
        gamma = math.sqrt(1-(1-b2)**i)/(1-(1-b1)**i)
        print "iteration {} cost {} parameters {} log update {} ".format(i, cost, unflatten(x), update)
        update = step_size*gamma*m/np.sqrt(i*v)
        if callback: callback(unflatten(x), i, unflatten(g))
        x = np.exp(np.log(x) - update)
        unflattened_x = unflatten(x)
        cost_list.append(cost)
        param1.append(unflattened_x[0])
        param2.append(unflattened_x[1])
        param3.append(unflattened_x[2])

    return cost_list,param1,param2,param3
