import torch
import torch.distributions as dist

from daphne import daphne

from primitives import funcprimitives,bindingVars,topologicalSort
from tests import is_tol, run_prob_test,load_truth

import matplotlib.pyplot as plt

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = { 'sqrt': torch.sqrt,
        '+': torch.add,
        '-': torch.sub,
        '/': torch.div,
        '*': torch.mul,
        'exp': torch.exp,
        '>': torch.greater,
        '<': torch.less,
        'normal': dist.Normal,
        'uniform': dist.Uniform,
        'exponential': dist.Exponential,
        'beta': dist.Beta,
        'discrete':dist.Categorical,
        'vector': torch.stack,
        'mat-transpose': torch.t,
        'mat-add': torch.add,
        'mat-mul': torch.matmul,
        'mat-repmat': None,
        'mat-tanh': torch.tanh,
        'get': None,
        'put': None,
        'first': None,
        'last': None,
        'append': None,
        'hash-map': None,
        'sample*': None,
        'observe*': None,
        'if': None}


def deterministic_eval(exp):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = env[exp[0]]
        args = list(map(deterministic_eval,exp[1:]))
        return funcprimitives(exp[0],op,args)
    elif type(exp) is int or type(exp) is float:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
    else:
        print(exp)
        raise("Expression type unknown.", exp)


def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    G = graph[1] #get the graph structure
    linkFuncs = G['P'] #get the associated link functions
    vertices  = G['V'] #get the list of vertices
    edges = G['A'] #get the list of edges
    E = graph[2] #get the expression we want to evaluate
    R = {}

    vertices = topologicalSort(vertices,edges)

    for i in vertices:
        linkFuncsEval = bindingVars(R,linkFuncs[i])
        R[i]  = deterministic_eval(linkFuncsEval)


    #there is an assumption that if the return expression E is singular
    #then it belongs to some node

    if type(E) == str:
        E = float(R[E]) #hacky, this just gets turned back into a tensor
    else:
        # find and replace the variables with the results from evaluating the link functions
        E = bindingVars(R,E)

    return deterministic_eval(E)


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)




#Testing:

def run_deterministic_tests():
    
    for i in range(1,13):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1])
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    #TODO: 
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        graph = daphne(['graph', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    

def plotResults(data):

    #want to turn data into a tensor object to easily extract peices
    #determine how many elements are in each entry
    e = data[0]
    
    #if each element in data is a list, i.e. a compounding list of objects
    if type(e) == list:
        #get the number of elements per entry
        numObjPerEntry = len(e)
        #create one object for each entry that contains all the samples
        for j in range(numObjPerEntry):
            tensorData = []
            for d in data:
                tensorData.append(d[j])

            #once all the samples are collected create a tensor object of them for easy access
            tensorData = torch.stack(tensorData)
            #determine the size of this object (should be vectors or matrices)
            tensorSize = tensorData.size()
            numRows = tensorSize[1]
            numCols = tensorSize[2]
            means = []
            stds  = []
            #create subplots of size of data
            fig, axes = plt.subplots(nrows=numRows, ncols=numCols, figsize=(10, 10))
            for r in range(numRows):
                for c in range(numCols):
                    if numRows == 1:
                        m = float(tensorData[:,r,c].mean())
                        s = float(tensorData[:,r,c].std())
                        means.append(float(tensorData[:,r,c].mean()))
                        stds.append(float(tensorData[:,r,c].std()))
                        ax = axes[c]
                        ax.hist(tensorData[:,r,c].numpy())
                        ax.set_xlabel('Samples')
                        ax.set_ylabel('PDF')
                        ax.set_title('mean = '+ "{:.4f}".format(m) + ' std = ' + "{:.4f}".format(s))
                    elif numCols == 1:
                        m = float(tensorData[:,r,c].mean())
                        s = float(tensorData[:,r,c].std())
                        means.append(float(tensorData[:,r,c].mean()))
                        stds.append(float(tensorData[:,r,c].std()))
                        ax = axes[r]
                        ax.hist(tensorData[:,r,c].numpy())
                        ax.set_xlabel('Samples')
                        ax.set_ylabel('PDF')
                        ax.set_title('mean = '+ "{:.4f}".format(m) + ' std = ' + "{:.4f}".format(s))
                    else:
                        m = float(tensorData[:,r,c].mean())
                        s = float(tensorData[:,r,c].std())
                        means.append(float(tensorData[:,r,c].mean()))
                        stds.append(float(tensorData[:,r,c].std()))
                        ax = axes[r,c]
                        ax.hist(tensorData[:,r,c].numpy())
                        ax.set_xlabel('Samples')
                        ax.set_ylabel('PDF')
                        ax.set_title('mean ='+ "{:.4f}".format(m) + ' std = ' + "{:.4f}".format(s))
            print('mean = ', means, 'std = ', stds)
            plt.show()

    else:
        tensorData = torch.stack(data)
        tensorSize = tensorData.size()
        #one dimensional dataset
        if(len(tensorSize) == 1):
            print('mean = ',float(tensorData.mean()),' std = ', float(tensorData.std()))
            m = float(tensorData.mean())
            s = float(tensorData.std())
            fig = plt.hist(tensorData.numpy(),bins=10)
            plt.xlabel('Samples')
            plt.ylabel('PDF')
            plt.title('mean ='+ "{:.4f}".format(m) + ' std = ' + "{:.4f}".format(s))
            plt.show()
        elif(len(tensorSize) == 2):
            numRows = 1
            numCols = tensorSize[1]
            means = []
            stds  = []
            fig, axes = plt.subplots(nrows=numRows, ncols=numCols, figsize=(10, 10))
            for r in range(numRows):
                for c in range(numCols):
                    #m = float(tensorData[:,c].mean())
                    #s = float(tensorData[:,c].std())
                    #means.append(float(tensorData[:,c].mean()))
                    #stds.append(float(tensorData[:,c].std()))
                    ax = axes[c]
                    ax.hist(tensorData[:,c].numpy())
                    ax.set_xlabel('Samples')
                    ax.set_ylabel('PDF')
                    ax.set_title('Step = '+ str(c))
                    #ax.set_title('mean ='+ "{:.4f}".format(m) + ' std = ' + "{:.4f}".format(s))
            print('mean = ', means, ' std = ', stds)
            plt.show()
        
        
if __name__ == '__main__':
    

    run_deterministic_tests()
    run_probabilistic_tests()

    num_samples = 1000
    samplesCollected = [[],[],[],[],[]]

    for i in range(1,5):
        graph = daphne(['graph','-i','../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        for j in range(num_samples):
            samplesCollected[i-1].append(sample_from_joint(graph))
        plotResults(samplesCollected[i-1])
        print(samplesCollected[i-1][0])    

    