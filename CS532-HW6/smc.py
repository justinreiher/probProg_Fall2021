from evaluator import evaluate
import torch
import numpy as np
import json
import sys

from timeit import default_timer as timer
from datetime import timedelta

import matplotlib.pyplot as plt
import seaborn as sns


def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = (res, None, {'done' : True}) #wrap it back up in a tuple, that has "done" in the sigma map
    return res

def resample_particles(particles, log_weights):

    logWeights = torch.stack(log_weights)
    Wnorm = logWeights.exp().sum()
    Ws = logWeights.exp()
    D = torch.distributions.Categorical(Ws/Wnorm)
    new_particles = particles.copy()
    for i in range(len(particles)):
        ind = int(D.sample())
        new_particles[i] = particles[ind]

    logZ = torch.log(1/len(logWeights) * Ws.sum())
    return logZ, new_particles



def SMC(n_particles, exp):

    particles = []
    weights = []
    logZs = []
    output = lambda x: x

    for i in range(n_particles):

        res = evaluate(exp, env=None)('addr_start', output)
        logW = 0.


        particles.append(res)
        weights.append(logW)

    #can't be done after the first step, under the address transform, so this should be fine:
    done = False
    smc_cnter = 0
    while not done:
    #    print('In SMC step {}, Zs: '.format(smc_cnter), logZs)
        for i in range(n_particles): #Even though this can be parallelized, we run it serially
            res = run_until_observe_or_end(particles[i])
            if 'done' in res[2]: #this checks if the calculation is done
                particles[i] = res[0]
                if i == 0:
                    done = True  #and enforces everything to be the same as the first particle
                    address = ''
                else:
                    if not done:
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:
                sig = res[2]
                if i == 0: # the first particle (we can get away with this because they run sequentially)
                    addrCurrent = sig['addr']
                else:
                    if not(sig['addr'] == addrCurrent):
                        raise RuntimeError('SMC Failed, particle ', i, 'arrived to observe at address: ', sig['addr'], ' which is different to the current address: ', addrCurrent)

                weights[i] = sig['logW']
                particles[i] = res

        if not done:
            #resample and keep track of logZs
            logZn, particles = resample_particles(particles, weights)
            logZs.append(logZn)
        smc_cnter += 1
    logZ = sum(logZs)
    return logZ, particles


if __name__ == '__main__':
    inc = [1,10,100,1000,10000,100000]
    for i in range(1,5):
        with open('programs/{}.json'.format(i),'r') as f:
            exp = json.load(f)

        fig, axes = plt.subplots(nrows = 1, ncols=len(inc),figsize=(10,10))

        for j in range(len(inc)):
            n_particles = inc[j]#None #TODO
            start = timer() 
            logZ, particles = SMC(n_particles, exp)
            end = timer()

            print("Elapsed time for program ", i,".daphne is: ",timedelta(seconds=end-start)," seconds with ",inc[j], " particles")
            print('logZ: ', logZ)

            if i == 3:
                dim = len(particles[0])
                values = torch.stack(particles)
                for k in range(dim):
                    print('Mean of dim',k,':',np.mean(values[:,k].numpy()))
                    print('Variance of dim',k,':',np.var(values[:,k].numpy()))

            else:
                particles = [torch.tensor(float(i)) for i in particles]
                values = torch.stack(particles)
            #TODO: some presentation of the results

                print('Mean of particles: ',values.mean())
                print('Variance of particles: ',values.var())


            ax = axes[j]
            ax.hist(values.numpy())
            ax.set_title('Program ' + str(i) + ' with ' + str(inc[j]) + ' particles')
            ax.set_xlabel('Particle Values')
            ax.set_ylabel('PDF Estimate')

        plt.show()
