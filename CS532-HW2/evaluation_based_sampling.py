from daphne import daphne
from tests import is_tol, run_prob_test,load_truth

import torch
import torch.distributions as dist
import matplotlib.pyplot as plt

#global state
sig = []
var = {}
rho = []

operations = {'sqrt': torch.sqrt,
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
              'hash-map': None}
        
def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    s = sig
    p = rho
    if p == []: #start with populating the procedure definitions, singleton like pattern used here
        for i in range(0,len(ast)):
            p.append(ast[i])
            ret,s = evaluate_program(p[i])
        p.clear() # after the program is run clear the procedure definitions
        s.clear() # reset the world state
        var.clear() # clear the variables
    else:
        e = ast

        if type(e) == int or type(e) == float:
            ret = torch.tensor(float(e))

        elif e[0] == 'if':
            testExp,s = evaluate_program(e[1])
            if testExp:
                ret,s = evaluate_program(e[2])
            else:
                ret,s = evaluate_program(e[3])

        elif e[0] == 'let':

            letBlock = e[1]
            var[letBlock[0]],s = evaluate_program(letBlock[1])
            ret,s = evaluate_program(e[2])
            del var[letBlock[0]] # get ride of the bound variable afterwards, it is no longer in scope after the let block is complete
        elif e[0] == 'sample':
            d = e[1] # get the distribution declaration
            try:
                Dobj,s = evaluate_program(d) # assign the distribution object
                if Dobj == []:
                    #variable not set, must be in a defn
                    ret = []
                else:
                    ret = Dobj.sample() #sample from that distribution object with the associated parameters
            except: #if the distribution object exists as a variable, then the parameters will have been set
                Dobj = var[d]
                ret = Dobj.sample()


        elif e[0] == 'observe':
            d = e[1] # get the distribution declaration
            y = e[2] # here I don't care about y in this assignment, but put here for the future
            try: #first try to get a distribution object from the known distributions directly
                 #otherwise the distribution object may be a variable
                Dobj,s = evaluate_program(d) #assign the distribution object
                if Dobj == []:
                    #variable not set, must be in a defn
                    ret = []
                else:
                    ret = Dobj.sample() #This is what we are doing now, but will be replaced later
            except:
                Dobj = var[d] #if retrieved from variable, then arguments already set
                ret = Dobj.sample()

        elif type(e) == list:
            args = []
            for i in range(1,len(e)):
                r,s = evaluate_program(e[i]) #only interested in the returned expression
                args.append(r) #wrap each experession as a program that needs to be unpacked

            try: #determine if this is a built in primitive function

                assert(e[0] in operations.keys())

                op = operations[e[0]] #retrieve a built-in function
                if e[0] == 'vector': #this is a constructor and so the arguments are not broadcast
                    try:
                        ret = op(args)
                    except:
                        ret = args
                elif e[0] == 'get': #this is a getter method
                    #arguments takes the form, (dict,index)
                    myDict = args[0]
                    if type(e[2]) == int:
                        ret = myDict[e[2]] #here we want the non-tensor (integer or original key to index the dict)
                    else:
                        ret = myDict[int(args[1])]
                elif e[0] == 'put': #this is a setter method
                    #arguments takes the form, (dict,index,value)
                    argsCopy = args.copy() # if we don't do this, then the original arguments will be modified!
                    myDict = argsCopy[0]
                    if type(e[2]) == int:
                        myDict[e[2]] = args[2]
                    else:
                        myDict[int(args[1])] = args[2]
                    ret = myDict   #return the new dictionary
                elif e[0] == 'first':
                    #arguments take the form, (vec), return the first element
                    vec = args[0]
                    ret = vec[0]
                elif e[0] == 'last':
                    #arguments take the form, (vec), return the last element
                    vec = args[0]
                    ret = vec[len(vec)-1]
                elif e[0] == 'append':
                    #arguments take the form, (vec,value)
                    vec = args[0]
                    ret = torch.cat((vec,torch.tensor([args[1]])),0)
                elif e[0] == 'hash-map':
                    myDict = {} #need to build the hash-map with non-tensor keys, want to make use of the arguments collected above
                    i = 1
                    while(i < len(e)):
                        myDict[e[i]] = args[i] # this is damn confusing, but e[1] corresponds to args[0], so e[1] is key 1 and args[1] is e[2] which is the value we want
                        i += 2
                    ret = myDict
                elif e[0] == 'mat-repmat':
                    ret = e[1].repeat(e[2],e[3])
                elif e[0] == 'mat-transpose':
                    ret = op(e[1])
                    print(ret)
                    input('make sure I get here')
                else:
                    ret = op(*args)

            except:   #if it does not match to a primitive function, then it must be a function definition
                if e[0] == 'defn':
                    #function definition format (defn func-name [func-args] [func-body] )
                    ret = None
                    for i in range(0,len(e[2])):
                        var[e[2][i]] = [] #place holder for the name of the arguments of the function
                        # this lambda expression binds variables to the constants in x and then runs the procedure
                    var[e[1]] = lambda x: [[var.update({e[2][i]:x[i]}) for i in range(0,len(x))], evaluate_program(e[3])] # the code for the function body
            
                else:
                    try:
                        # does the function already exists?
                        [_,(ret,s)] = var[e[0]](args)
                    # if not evaluate the program 
                    except:             
                        # this fall through is annoying and I would like it to be avoided really, not sure what to do
                        ret = []

        # if there is nothing to pattern match against, then it must be a variable
        else:
        
            try:
                #if the variable has a binding return the binding
                ret = var[e]
            except:
                var[e] = []
                ret = var[e]

    return ret,s


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)[0]
    


def run_deterministic_tests():
    
    for i in range(14,14):
         #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))

        ret, sig = evaluate_program(ast)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(7,7):
        #note: this path should be with respect to the daphne path!        
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        stream = get_stream(ast)
        
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
            #create subplots of size of data
            fig, axes = plt.subplots(nrows=numRows, ncols=numCols, figsize=(10, 10))
            for r in range(numRows):
                for c in range(numCols):
                    if numRows == 1:
                        ax = axes[c]
                        ax.hist(tensorData[:,r,c].numpy())
                    elif numCols == 1:
                        ax = axes[r]
                        ax.hist(tensorData[:,r,c].numpy())
                    else:
                        ax = axes[r,c]
                        ax.hist(tensorData[:,r,c].numpy())
            plt.show()

    else:
        tensorData = torch.stack(data)
        tensorSize = tensorData.size()
        #one dimensional dataset
        if(len(tensorSize) == 1):
            fig = plt.hist(tensorData.numpy(),bins=10)
            plt.show()
        elif(len(tensorSize) == 2):
            numRows = 1
            numCols = tensorSize[1]
            fig, axes = plt.subplots(nrows=numRows, ncols=numCols, figsize=(10, 10))
            for r in range(numRows):
                for c in range(numCols):
                    ax = axes[c]
                    ax.hist(tensorData[:,c].numpy())
            plt.show()

        
if __name__ == '__main__':

    run_deterministic_tests()
    
    run_probabilistic_tests()

    num_samples = 1000
    samplesCollected = [[],[],[],[],[]]

    for i in range(1,5):
        ast = daphne(['desugar', '-i', '../CS532-HW2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        for j in range(0,num_samples):
            samplesCollected[i-1].append(evaluate_program(ast)[0])
        plotResults(samplesCollected[i-1])
        print(samplesCollected[i-1][0])

