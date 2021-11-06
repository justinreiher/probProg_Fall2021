import torch

from daphne import daphne
from bbvi import bbvi
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
from plottingRoutine import plotVariationalData

def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield vvbi(graph)

def plotBBVI(X,pNum):
    
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

def weightSamples(samples,weights,pNum):
    r,c = weights.shape
    ret = []
    elbo = []
    if pNum == 4:
        b = torch.exp(torch.tensor(1))
        for i in range(r):
            s_i = samples[i]
            logw_i = weights[i,:]
            Wnorm = torch.float_power(b,logw_i).sum()
            elbo.append(logw_i.mean())
            weightedSamples = []
            for j in range(c):
                weightedSamples.append([])
                for k in range(len(s_i[j])):
                    weightedSamples[j].append(s_i[j][k]*(torch.float_power(b,logw_i[j])/Wnorm))
            colItems = []
            for k in range(len(s_i[0])):
                retlist = []
                for j in range(c):
                    retlist.append(weightedSamples[j][k])
                colItems.append(torch.stack(retlist).sum(0))
            ret.append(colItems)

    else:
        for i in range(r):
            s_i = samples[i]
            logw_i = weights[i,:]
            Wnorm = logw_i.exp().sum()
            elbo.append(logw_i.mean())
            weightedSamples = []
            for j in range(c):
                weightedSamples.append(s_i[j]*logw_i[j].exp()/Wnorm)
            ret.append(torch.stack(weightedSamples).sum(0))

    return ret,torch.stack(elbo)




if __name__ == '__main__':
    

   # num_samples = [200,200,25,100,25]
   # T = [350,350,350,150,350]
    num_samples = [200,200,25,200,100]
    T = [350,350,350,150,350]
    lr = [0.5,0.25,0.05,0.05,0.05]
    for i in range(5,6):
        graph = daphne(['graph','-i','../CS532-HW4/programs/{}.daphne'.format(i)])
        print('\n\n\nCollect samples denoted by program {}:'.format(i))
        start = timer()
        s,logWeights,Qf,Qps = bbvi(graph,T[i-1],num_samples[i-1],lr[i-1])
        end = timer()
        print('\n\n\nCollect samples denoted by program {}:'.format(i))
        print("Elapsed time for program ", i,".daphne is: ",timedelta(seconds=end-start)," seconds")
        weightedSamples,elbo = weightSamples(s,logWeights,i)
        if i == 4:
            plotVariationalData(Qf,weightedSamples,i)
        else:
            print('Mean of samples: ',torch.stack(weightedSamples).mean(0))
            print('and variance of samples: ', torch.stack(weightedSamples).var(0))
            plotVariationalData(Qf,Qps,i)
        plt.plot(elbo.numpy())
        plt.title('ELBO ' + ' T='+str(T[i-1]) +' L='+str(num_samples[i-1]) +' Adam lr='+str(lr[i-1]))
        plt.xlabel('Step #')
        plt.ylabel('Log Weight Mean')
        plt.show()
  #      input("")       
  #      plotResults(samplesCollected[i-1])
  #      print(samplesCollected)
  #      plotBBVI(samplesCollected[i-1],i)   

    