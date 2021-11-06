import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plotVariationalData(Qf,Qps,pNum):
    if pNum == 1:
        plotVariationalDataP1(Qf,Qps)
    elif pNum == 2:
        plotVariationalDataP2(Qf,Qps)
    elif pNum == 3:
        plotVariationalDataP3(Qf,Qps)
    elif pNum == 4:
        plotVariationalDataP4(Qf,Qps)
    else:
        plotVariationalDataP5(Qf,Qps)

def plotVariationalDataP1(Qf,Qps):

    numSamples = 3000
    qSamples = []
    for q in Qf:
        sq = []
        for i in range(numSamples):
            sq.append(Qf[q].sample())
        qSamples.append(sq)

    plt.hist(torch.stack(qSamples[0]).numpy())
    plt.xlabel('Samples')
    plt.ylabel('PDF Estimate')
    plt.title('Distribution of mu')
    plt.show()

    p = []
    for q in Qps:
        params = torch.stack(Qps[q])
        if(len(params) > 1):
            r,c = params.shape
            for i in range(c):
                p.append(params[:,i])

    fig, axes = plt.subplots(nrows = len(p), ncols=1,figsize=(10,10))
    titles = ['mu','sigma']
    
    for i in range(len(p)):
        ax = axes[i]
        ax.plot(p[i])
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Param')
        ax.set_title(titles[i])

    plt.show()

def plotVariationalDataP2(Qf,Qps):

    numSamples = 3000
    qSamples = []
    for q in Qf:
        sq = []
        for i in range(numSamples):
            sq.append(Qf[q].sample())
        qSamples.append(sq)

    title = ['slope','bias']

    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize=(10,10))
    axes[0].hist(torch.stack(qSamples[0]).numpy())
    axes[0].set_title('Slope Distribution Estimate')
    axes[0].set_xlabel('Samples')
    axes[0].set_ylabel('PDF Estimate')

    axes[1].hist(torch.stack(qSamples[1]).numpy())
    axes[1].set_title('Bias Distribution Estimate')
    axes[1].set_xlabel('Samples')
    axes[1].set_ylabel('PDF Estimate')
    plt.show()

    p = []
    for q in Qps:
        params = torch.stack(Qps[q])
        if(len(params) > 1):
            r,c = params.shape
            for i in range(c):
                p.append(params[:,i])

    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(10,10))
    
    axes[0][0].plot(p[0])
    axes[0][0].set_title('Slope Mean Estimate')
    axes[0][0].set_xlabel('Iterations')
    axes[0][0].set_ylabel('Param Estimate')

    axes[1][0].plot(p[1])
    axes[1][0].set_title('Slope Sigma Estimate')
    axes[1][0].set_xlabel('Iterations')
    axes[1][0].set_ylabel('Param Estimate')

    axes[0][1].plot(p[2])
    axes[0][1].set_title('Bias Mean Estimate')
    axes[0][1].set_xlabel('Iterations')
    axes[0][1].set_ylabel('Param Estimate')

    axes[1][1].plot(p[3])
    axes[1][1].set_title('Bias Sigma Estimate')
    axes[1][1].set_xlabel('Iterations')
    axes[1][1].set_ylabel('Param Estimate')

    plt.show()

def plotVariationalDataP3(Qf,Qps):

    numSamples = 3000
    qSamplesMu = []
    qSamplesSig = []
    qSamplesPi = []

    for q in Qf:
        sq = []
        if q == 'sample0' or q == 'sample2' or q == 'sample4':
            for i in range(numSamples):
                sq.append(Qf[q].sample())
            qSamplesMu.append(sq)
        elif q == 'sample1' or q == 'sample3' or q == 'sample5':
            for i in range(numSamples):
                sq.append(Qf[q].sample())
            qSamplesSig.append(sq)
        else:
            for i in range(numSamples):
                sq.append(Qf[q].sample())
            qSamplesPi.append(sq)

    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize=(10,10))
    axes[0].hist(torch.stack(qSamplesMu[0]).numpy())
    axes[0].hist(torch.stack(qSamplesMu[1]).numpy())
    axes[0].hist(torch.stack(qSamplesMu[2]).numpy())
    axes[0].set_title('Mu Distribution Estimate')
    axes[0].set_xlabel('Samples')
    axes[0].set_ylabel('PDF Estimate')

    axes[1].hist(torch.stack(qSamplesSig[0]).numpy())
    axes[1].hist(torch.stack(qSamplesSig[1]).numpy())
    axes[1].hist(torch.stack(qSamplesSig[2]).numpy())
    axes[1].set_title('Sigma Distribution Estimate')
    axes[1].set_xlabel('Samples')
    axes[1].set_ylabel('PDF Estimate')

    axes[2].hist(torch.stack(qSamplesPi[0]).numpy())
    axes[2].hist(torch.stack(qSamplesPi[1]).numpy())
    axes[2].hist(torch.stack(qSamplesPi[2]).numpy())
    axes[2].set_title('Pi Distribution Estimate')
    axes[2].set_xlabel('Samples')
    axes[2].set_ylabel('PDF Estimate')
    
    plt.show()
def plotVariationalDataP4(Qf,samples):
    numIters = len(samples)
    numItemsPerIter = len(samples[0])
    W0 = []
    b0 = []
    W1 = []
    b1 = []
    var = ['W0','b0','W1','b1']
    for i in range(numIters):
        W0.append(samples[i][0])
        b0.append(samples[i][1])
        W1.append(samples[i][2])
        b1.append(samples[i][3])

    s = [W0,b0,W1,b1]
    for i in range(numItemsPerIter):
        print('Mean of samples ',var[i],': ',torch.stack(s[i]).mean(0))
        print('and variance of samples ',var[i],': ',torch.stack(s[i]).var(0))

    ax = sns.heatmap(torch.stack(W0).mean(0).numpy(),linewidth=0.5)
    ax.set_title('W0')
    plt.show()
    ax = sns.heatmap(torch.stack(b0).mean(0).numpy(),linewidth=0.5)
    ax.set_title('b0')
    plt.show()
    ax = sns.heatmap(torch.stack(W1).mean(0).numpy(),linewidth=0.5)
    ax.set_title('W1')
    plt.show()
    ax = sns.heatmap(torch.stack(b1).mean(0).numpy(),linewidth=0.5)
    ax.set_title('b1')
    plt.show()

def plotVariationalDataP5(Qf,Qps):
    print(Qf)

    numSamples = 3000
    pdfEst = []
    for i in range(numSamples):
        pdfEst.append(Qf['sample2'].sample())

    Phist = Qps['sample2']
    plt.plot(torch.stack(Phist).numpy())
    plt.title('Parameters for s per Iteration')
    plt.xlabel('Iteration #')
    plt.ylabel('Parameters')
    plt.show()

    plt.hist(torch.stack(pdfEst).numpy())
    plt.title('Estimate of Distribution for s')
    plt.xlabel('Samples')
    plt.ylabel('PDF Estimate')
    plt.show()

