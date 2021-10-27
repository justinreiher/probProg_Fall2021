import torch

from daphne import daphne
from gibbs import gibbsMH
import matplotlib.pyplot as plt


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield gibbs(graph)
         
        
if __name__ == '__main__':
    

    num_samples = 5000
    samplesCollected = []

    for i in range(1,6):
        graph = daphne(['graph','-i','../CS532-HW3/programs/{}.daphne'.format(i)])
        print('\n\n\nCollect samples denoted by program {}:'.format(i))
        
        samplesCollected.append(gibbsMH(graph,num_samples))
  #      plotResults(samplesCollected[i-1])
  #      print(samplesCollected)
        print(torch.stack(samplesCollected[i-1]).mean(0))
      #  print(samplesCollected[i-1][num_samples-1])
        plt.plot(torch.stack(samplesCollected[i-1]).numpy())
        plt.show()   

    