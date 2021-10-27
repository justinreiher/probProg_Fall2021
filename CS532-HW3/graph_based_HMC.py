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
    

    num_samples = 5000
    T = 20
    eps = 0.01
    samplesCollected = []
    j = 0

    for i in [1,2,5]:
        graph = daphne(['graph','-i','../CS532-HW3/programs/{}.daphne'.format(i)])
        print('\n\n\nCollect samples denoted by program {}:'.format(i))
        
        samplesCollected.append(HMC(graph,num_samples,T,eps))
  #      plotResults(samplesCollected[i-1])
  #      print(samplesCollected)
        print(torch.stack(samplesCollected[j]).mean(0))
      #  print(samplesCollected[i-1][num_samples-1])
        plt.plot(torch.stack(samplesCollected[j]).detach().numpy())
        plt.show()
        j += 1   

    