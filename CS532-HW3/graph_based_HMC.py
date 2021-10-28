import torch

from daphne import daphne
from HMC import HMC
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np

def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield HMC(graph)

def plotHMC(X,pNum):
    
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
    

    num_samples = 5000
    T = 20
    eps = 0.01
    samplesCollected = []
    j = 0

    for i in [1,2,5]:
        graph = daphne(['graph','-i','../CS532-HW3/programs/{}.daphne'.format(i)])
        print('\n\n\nCollect samples denoted by program {}:'.format(i))
        if i == 5:
            eps = 0.1
        start = timer()
        s,jll = HMC(graph,num_samples,T,eps)
        samplesCollected.append(s)
        end = timer()
        print("Elapsed time for program ", i,".daphne is: ",timedelta(seconds=end-start)," seconds")
        plt.plot(torch.stack(jll).detach().numpy())
        plt.title('Joint Log Likelihood')
        plt.xlabel('Sample #')
        plt.ylabel('Joint Log Likelihood')

        print(torch.stack(samplesCollected[j]).mean(0))
      #  print(samplesCollected[i-1][num_samples-1])
        plotHMC(samplesCollected[j],i)
        j += 1   

    