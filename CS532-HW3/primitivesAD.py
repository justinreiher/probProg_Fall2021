import torch
import torch.distributions as dist

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
        '=': torch.equal,
        'or': None,
        'and': None,
        'true': torch.tensor(1.0),
        'false': torch.tensor(0.0),
        'normal': dist.Normal,
        'uniform': dist.Uniform,
        'exponential': dist.Exponential,
        'beta': dist.Beta,
        'discrete': dist.Categorical,
        'dirichlet': dist.Dirichlet,
        'flip': dist.Bernoulli,
        'gamma': dist.Gamma,
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
    elif type(exp) is int or type(exp) is float or type(exp) is bool:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp),requires_grad=True)
    elif type(exp) is type(torch.tensor(1)):
    	return exp #already a tensor object
    elif exp == None:
    	return torch.tensor(float(0))
    else:
        print(exp)
        raise("Expression type unknown.", exp)

def funcprimitives(e,op,args):
	if e == 'vector':
		try:
			ret = op(args)
		except:
			ret = args
	elif e == 'sample*':
		Dobj = args[0]
		ret = Dobj.sample()
	elif e == 'observe*':
		Dobj = args[0]
		ret = Dobj.log_prob(args[1])
	elif e == 'if':
		if args[0]:
			ret = args[1]
		else:
			ret = args[2]
	elif e == 'and':
		ret = args[0] and args[1]
	elif e == 'or':
		ret = args[0] or args[1]
	elif e == 'rest':
		ret = args[0][1:]
	elif e == 'get':
		ind = int(args[1])
		ret = args[0][ind]
	elif e == 'mat-transpose':
		ret = op(args[0])
	elif e == 'mat-repmat':
		ret = args[0].repeat(int(args[1]),int(args[2]))
	elif e == 'mat-mul':
		ret = op(*args)
	elif e == '=':
		ret = torch.tensor(float(op(*args)))
	elif e == 'dirac':
		sigma = 0.1
		ret = dist.Normal(args[0],sigma)
	else:
		ret = op(*args)

	return ret

def bindingVars(varDict,funcCode):
	evalFuncCode = funcCode.copy()
	for i in range(0,len(funcCode)):
		if type(evalFuncCode[i]) == list:
			evalFuncCode[i] = bindingVars(varDict,evalFuncCode[i])
		elif evalFuncCode[i] in varDict.keys():
			evalFuncCode[i] = varDict[evalFuncCode[i]]
	return evalFuncCode

def topologicalSort(listOfVertices,listOfEdges):
	#implementation of Khan's algorithm from:
	#https://www.geeksforgeeks.org/topological-sorting-indegree-based-solution/
	indegreeEdges = {}
	sortedNodes = []
	queue = []
	nodesVisited = 0
	#create dictionary of all the nodes and the number of ingoing edges
	for n in listOfVertices:
		indegreeEdges[n] = 0
	# go through all the nodes that have incoming edges and increment the
	# dictionary accordingly
	for n in listOfEdges:
		edges = listOfEdges[n]
		for e in edges:
			indegreeEdges[e] += 1

	# Find all nodes that have no incoming edges and put them in a queue
	for n in indegreeEdges:
		if indegreeEdges[n] == 0:
			queue.append(n)

	# if the queue starts empty, it means that there are no vertices that do not 
	# have an incoming edge, in this case return the list of nodes
	if queue == []:
		return listOfVertices

	# while the queue is not empty, take a node off the queue (it has no incoming edges if it's on the queue)
	# add the node to the list of sorted nodes, increment the number of nodes visited,
	# visit the child nodes given from listOfEdges (if there are any)
	# when visiting a node decrement the indegree by 1 and check if it is 0, if so add it to the queue
	while(queue != []):
		n = queue.pop()
		nodesVisited += 1
		#if node n has neighbours get them, otherwise there are no neighbours to visit
		if n in listOfEdges.keys():
			neighbours = listOfEdges[n]
		else:
			neighbours = []
		sortedNodes.append(n)
		for v in neighbours:
			indegreeEdges[v] -= 1
			if indegreeEdges[v] == 0:
				queue.append(v)

	#verify that we have visited every node, otherwise something has gone wrong
	assert(nodesVisited == len(listOfVertices))

	return sortedNodes

def childParentMapping(vertices,edges,P):

	for i in vertices:
		for j in edges:
			if(i in edges[j]):
				listOfP = P[i]
				#want this to be a set, so no duplication
				if not(j in listOfP):
					listOfP.append(j)
	return P
