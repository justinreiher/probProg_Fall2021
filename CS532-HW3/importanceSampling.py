from daphne import daphne

import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
from eval import evaluate_program
from timeit import default_timer as timer
from datetime import timedelta

import numpy as np


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)[0]
    


def likelihoodWeighting(L,ast):
    ret = []
    w = []
    for l in range(L):
        r,s = evaluate_program(ast)
        ret.append(r)
        w.append(s)
    return ret,w

def plotIS(r,w,pNum):
    x = []
    for i in range(len(r)):
        x.append(w[i].exp()*r[i])
    Wnorm = torch.stack(w).exp().cumsum(0)
    x = torch.stack(x).cumsum(0)
    X = []
    for i in range(len(x)):
        res = x[i]/Wnorm[i]
        #remove any samples that return nan
        if(not(torch.any(torch.isnan(res)))):
            X.append(res)
    X = torch.stack(X)
    print(X)
    [sig2,mu] = torch.var_mean(X,0)
    print('Mean of trace: ', mu ,' and variance of trace: ',sig2)
    if pNum == 2:
        print(X.t().numpy())
        print('The covariance matrix: ', np.cov(X.t().numpy()))

    if pNum == 2 or pNum == 5:
        fig,axes = plt.subplots(nrows=2,ncols=2)
        axes[0][0].hist(X[:,0].numpy())
        axes[0][0].set_xlabel('Sample #')
        axes[0][0].set_ylabel('PDF Estimate')
        axes[0][0].set_title('Program ' +  str(pNum) + ' with mean: ' +"{:.4}".format(float(mu[0])) + ' variance: '+"{:.4}".format(float(sig2[0])))

        axes[1][0].plot(X[:,0].numpy())
        axes[1][0].set_xlabel('Sample #')
        axes[1][0].set_ylabel('Sample Values')
        axes[1][0].set_title('Samples of Variable 0 in program: '+str(pNum))

        axes[0][1].hist(X[:,1].numpy())
        axes[0][1].set_xlabel('Samples')
        axes[0][1].set_ylabel('PDF Estimate')
        axes[0][1].set_title('Program ' +  str(pNum) + ' with mean: ' +"{:.4}".format(float(mu[1])) + ' variance: '+"{:.4}".format(float(sig2[1])))

        axes[1][1].plot(X[:,1].numpy())
        axes[1][1].set_xlabel('Sample #')
        axes[1][1].set_ylabel('Sample Values')
        axes[1][1].set_title('Samples of Variable 1 in program: '+str(pNum))

    else: #program 1 which is the Gaussian with observes 7 and 8
        fig,axes = plt.subplots(nrows = 2, ncols = 1)
        axes[0].hist(X.numpy())
        axes[0].set_xlabel('Sample #')
        axes[0].set_ylabel('PDF Estimate')
        axes[0].set_title('Program ' + str(pNum) + ' with mean: ' +"{:.4}".format(float(mu)) + ' variance: ' +"{:.4}".format(float(sig2)))
        axes[1].plot(X.numpy())
        axes[1].set_xlabel('Sample #')
        axes[1].set_ylabel('Sample Values')
        axes[1].set_title('Samples of program: ' + str(pNum))

    plt.show()
        
if __name__ == '__main__':


    num_samples =30000
    
    for i in range(1,6):
        ast = daphne(['desugar', '-i', '../CS532-HW3/programs/{}.daphne'.format(i)])
        print('\n\n\nCollect samples denoted by program {}:'.format(i))
        start = timer()
        r,w = likelihoodWeighting(num_samples,ast)
        end = timer()
        print("Elapsed time for program ", i,".daphne is: ",timedelta(seconds=end-start)," seconds")
        plt.plot(torch.stack(w).numpy())
        plt.title('Joint Log Likelihood')
        plt.xlabel('Sample #')
        plt.ylabel('Joint Log Likelihood') 
        Wnorm = torch.stack(w).exp().sum()
        x =[]
        for j in range(len(r)):
            x.append(w[j].exp()*r[j]/Wnorm)
       # plotResults(result)
        print(sum(x))
        pNum = i
        plotIS(r,w,pNum)


