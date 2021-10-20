import torch
import torch.distributions as dist

from daphne import daphne

from primitives import funcprimitives,bindingVars,topologicalSort
from tests import is_tol, run_prob_test,load_truth

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
      #  input('trying a different approach')
      #  for i in range(0,len(E)):
      #      if E[i] in R.keys():
      #          E[i] = float(R[E[i]])

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


        
        
if __name__ == '__main__':
    

    run_deterministic_tests()
    run_probabilistic_tests()




    for i in range(1,5):
        graph = daphne(['graph','-i','../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(sample_from_joint(graph))    

    