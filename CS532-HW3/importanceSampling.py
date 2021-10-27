from daphne import daphne

import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
from eval import evaluate_program



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

def plotIS(r,w):
    x = []
    for i in range(len(r)):
        x.append(w[i].exp()*r[i])
    Wnorm = torch.stack(w).exp().cumsum(0)
    x = torch.stack(x).cumsum(0)

    X = []
    for i in range(len(x)):
        X.append(x[i]/Wnorm[i])
    plt.plot(torch.stack(X).numpy())
    plt.show()
        
if __name__ == '__main__':


    num_samples =10000
    
    for i in range(1,6):
        ast = daphne(['desugar', '-i', '../CS532-HW3/programs/{}.daphne'.format(i)])
        print('\n\n\nCollect samples denoted by program {}:'.format(i))
        r,w = likelihoodWeighting(num_samples,ast)

        Wnorm = torch.stack(w).exp().sum()
        x =[]
        for i in range(len(r)):
            x.append(w[i].exp()*r[i]/Wnorm)
       # plotResults(result)
        print(sum(x))
        plotIS(r,w)


