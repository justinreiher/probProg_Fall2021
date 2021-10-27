import torch

from daphne import daphne
from HMC import HMC
import matplotlib.pyplot as plt


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield HMC(graph)
         
        
if __name__ == '__main__':
    

    num_samples = 2000
    T = 15
    eps = 0.01
    samplesCollected = []

    for i in [1,2,5]:
        graph = daphne(['graph','-i','../CS532-HW3/programs/{}.daphne'.format(i)])
        print('\n\n\nCollect samples denoted by program {}:'.format(i))
        
        samplesCollected.append(HMC(graph,num_samples,T,eps))
  #      plotResults(samplesCollected[i-1])
  #      print(samplesCollected)
        print(torch.stack(samplesCollected[i-1]).mean(0))
      #  print(samplesCollected[i-1][num_samples-1])
        plt.plot(torch.stack(samplesCollected[i-1]).detach().numpy())
        plt.show()   

    