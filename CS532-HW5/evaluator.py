from primitives import env as penv
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from pyrsistent import pmap,plist
import torch

from timeit import default_timer as timer
from datetime import timedelta
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(5000)

Symbol = str
Number = (int, float, bool)
List = list
Atom = (Symbol,Number)
Exp = (Atom,List)

def standard_env():
    "An environment with some Scheme standard procedures."
    env = pmap(penv)
    env = env.update({'alpha' : ''}) 

    return env

class Env(dict):
    "An environment: a dictionary of ('var':val) paris with and outer Env."
    def __init__(self,params=(),args=(),outer=None):
        self.update(zip(params,args))
        self.outer = outer
    def get(self,var):
        "Find the innermost Env where var appears."
        return self[var] if (var in self) else self.outer.get(var)

class Procedure(object):
    "A user-defined FOPPL procedure."
    def __init__(self,params,body,env):
        self.params, self.body, self.env = params,body,env
    def __call__(self,*args):
        return evaluate(self.body, Env(self.params, args, self.env))



def evaluate(exp, env=None): #TODO: add sigma, or something
    # if the environment is not set, then get the standard environment, and add
    # sigma to this environment
    if env is None:
        env = standard_env()
        env = env.update({'sig':''})

    if isinstance(exp,Symbol):  #variable reference
        e = env.get(exp)
        if e == None:
            e = exp
        return e
    elif not isinstance(exp,List): #constant case
        return torch.tensor(float(exp))

    op, *args = exp
    if op == 'if':
        (test,conseq,alt) = args
        exp = (conseq if evaluate(test,env) else alt)
        return evaluate(exp,env)
    elif op == 'fn': #procedure definition
        (params,body) = args
        return Procedure(params,body,env)
    elif op == 'sample':
        v = evaluate(args[0],env)
        d = evaluate(args[1],env)
        return d.sample()
    elif op == 'observe':
        v = evaluate(args[0],env)
        d = evaluate(args[1],env)
        c = evaluate(args[2],env)
        return c
    else:
        proc = evaluate(op,env)
        vals = [evaluate(arg,env) for arg in args]
        return proc(*vals)

    return    


def get_stream(exp):
    while True:
        yield evaluate(exp)('0_')


def run_deterministic_tests():
    
    for i in range(1,14):

        exp = daphne(['desugar-hoppl', '-i', '../CS532-HW5/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)('0_')
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('FOPPL Tests passed')
        
    for i in range(1,13):

        exp = daphne(['desugar-hoppl', '-i', '../CS532-HW5/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)('0_')
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-2
    
    for i in range(1,7):
        exp = daphne(['desugar-hoppl', '-i', '../CS532-HW5/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(exp)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    



if __name__ == '__main__':
    
    run_deterministic_tests()
    run_probabilistic_tests()
    
    num_samples = 10000

    for i in range(1,4):
        print(i)
        exp = daphne(['desugar-hoppl', '-i', '../CS532-HW5/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        res = []
        start = timer()
        for j in range(num_samples):
            res.append(evaluate(exp)('0_'))
        end = timer()
        print("Elapsed time for program ", i,".daphne is: ",timedelta(seconds=end-start)," seconds")
        plt.hist(torch.stack(res).numpy())
        plt.title('Output for program ' + str(i))
        plt.xlabel('Samples')
        plt.ylabel('Estimate of PDF')
        if not(i == 3):
            print('Mean of samples: ', torch.stack(res).mean(0))
            print('Variance of samples: ', torch.stack(res).var(0))
        plt.show()



