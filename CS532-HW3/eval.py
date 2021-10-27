import torch
import torch.distributions as dist

#global state
sig = 0
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
              '=': torch.equal,
              'or': None,
              'and': None,
              'true': torch.tensor(1.0),
              'false': torch.tensor(0.0),
              'normal': dist.Normal,
              'uniform': dist.Uniform,
              'exponential': dist.Exponential,
              'beta': dist.Beta,
              'discrete':dist.Categorical,
              'dirichlet':dist.Dirichlet,
              'flip':dist.Bernoulli,
              'gamma':dist.Gamma,
              'dirac': None,
              'vector': torch.stack,
              'mat-transpose': torch.t,
              'mat-add': torch.add,
              'mat-mul': torch.matmul,
              'mat-repmat': None,
              'mat-tanh': torch.tanh,
              'get': None,
              'put': None,
              'first': None,
              'second': None,
              'rest': None,
              'last': None,
              'append': None,
              'hash-map': None}

def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    global sig 

    p = rho
    if p == []: #start with populating the procedure definitions, singleton like pattern used here
        sig = 0
        for i in range(0,len(ast)):
            p.append(ast[i])
            ret,sig = evaluate_program(p[i])
        p.clear() # after the program is run clear the procedure definitions
        var.clear() # clear the variables
    else:
        e = ast

        if type(e) == int or type(e) == float:
            ret = torch.tensor(float(e))
        elif type(e) == bool:
            #because Bernoulli distributions return float values 1.0 or 0.0 for True or False
            #this hack is required...
            if(e):
                ret = torch.tensor(1.0)
            else:
                ret = torch.tensor(0.0)

        elif e[0] == 'if':
            testExp,sig = evaluate_program(e[1])
            if testExp:
                ret,sig = evaluate_program(e[2])
            else:
                ret,sig = evaluate_program(e[3])

        elif e[0] == 'let':
            letBlock = e[1]
            var[letBlock[0]],sig = evaluate_program(letBlock[1])
            ret,sig = evaluate_program(e[2])
            del var[letBlock[0]] # get ride of the bound variable afterwards, it is no longer in scope after the let block is complete
        elif e[0] == 'sample':
            d = e[1] # get the distribution declaration
            try:
                Dobj,sig = evaluate_program(d) # assign the distribution object
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

                Dobj,sig = evaluate_program(d) #assign the distribution object
                ret,sig = evaluate_program(y)

                if Dobj == [] or y == []:
                    #variable not set, must be in a defn
                    ret = []
                else:
                    dSample = Dobj.sample() #Sample from the prior
                    sig = sig + Dobj.log_prob(ret)

            except:
                Dobj = var[d] #if retrieved from variable, then arguments already set
                ret = Dobj.sample()
                sig = sig + Dobj.log_prob(ret)

        elif type(e) == list:
            args = []
            for i in range(1,len(e)):
                r,sig = evaluate_program(e[i]) #only interested in the returned expression
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
                elif e[0] == 'second':
                    vec = args[0]
                    ret = vec[1]
                elif e[0] == 'rest':
                    vec = args[0]
                    ret = vec[1:]
                elif e[0] == 'last':
                    #arguments take the form, (vec), return the last element
                    vec = args[0]
                    ret = vec[len(vec)-1]
                elif e[0] == 'append':
                    #arguments take the form, (vec,value)
                    vec = args[0]
                    ret = torch.cat((vec,torch.tensor([args[1]])),0)
                elif e[0] == 'dirac':
                    sigma = 0.1
                    ret = dist.Normal(args[0],sigma)
                elif e[0] == 'hash-map':
                    myDict = {} #need to build the hash-map with non-tensor keys, want to make use of the arguments collected above
                    i = 1
                    while(i < len(e)):
                        myDict[e[i]] = args[i] # this is damn confusing, but e[1] corresponds to args[0], so e[1] is key 1 and args[1] is e[2] which is the value we want
                        i += 2
                    ret = myDict
                elif e[0] == 'mat-repmat':
                    ret = args[0].repeat(int(args[1]),int(args[2]))
                elif e[0] == 'mat-transpose':
                    ret = op(args[0])
                elif e[0] == 'true':
                    ret = op
                elif e[0] == 'false': 
                    ret = op
                elif e[0] == 'and':
                    ret = args[0] and args[1]
                elif e[0] == 'or':
                    ret = args[0] or args[1]
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
                        [_,(ret,sig)] = var[e[0]](args)
                    # if not evaluate the program 
                    except:             
                        # this fall through is annoying and I would like it to be avoided really, not sure what to do
                        ret = []
                        #print('fall through')
                        #print(e)
                        #input('oops')

        # if there is nothing to pattern match against, then it must be a variable
        else:
        
            try:
                #if the variable has a binding return the binding
                ret = var[e]
            except:
                var[e] = []
                ret = var[e]

    return ret,sig