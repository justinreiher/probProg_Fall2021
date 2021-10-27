import torch

from primitives import deterministic_eval,topologicalSort,childParentMapping,bindingVars
import torch.distributions as dist

G = []
linkFuncs = []
vertices = []
edges = []
Y = []
P = {}
Q = {}

def accept(x,Xp,X):
    linkEvalX = bindingVars(X,Q[x])
    DobjX = deterministic_eval(linkEvalX)
    linkEvalXp = bindingVars(Xp,Q[x])
    DobjXp = deterministic_eval(linkEvalXp)

    logAlpha = DobjXp.log_prob(X[x]) - DobjX.log_prob(Xp[x])
    markovBlanket = getMarkovBlanket(x)

    for v in markovBlanket:
    	linkEvalX = bindingVars(Xp,linkFuncs[v])
    	linkEvalXp = bindingVars(X,linkFuncs[v])

    	if linkEvalX[0] == 'sample*':
    		linkEvalX[0] = 'observe*'
    		linkEvalX.append(Xp[v])
    	if linkEvalXp[0] == 'sample*':
    		linkEvalXp[0] = 'observe*'
    		linkEvalXp.append(X[v])

    	logAlpha = logAlpha + deterministic_eval(linkEvalX)
    	logAlpha = logAlpha - deterministic_eval(linkEvalXp)

    return torch.exp(logAlpha)

def gibbs_step(X):
    for x in X:
    	linkEval = bindingVars(X,Q[x])
    	Dobj = deterministic_eval(linkEval)
    	Xp = X.copy()
    	Xp[x]  = Dobj.sample()
    	alpha = accept(x,Xp,X)
    	u  = dist.Uniform(0,1).sample()
    	if u < alpha:
    		X = Xp.copy()
    return X


def gibbsMH(graph,num_samples):
    "This function does Metropolis-Hastings within Gibbs"
    #set the global context
    global G
    global linkFuncs
    global vertices
    global edges
    global Y 
    global P
    global Q
    G = graph[1] #get the graph structure
    linkFuncs = G['P'] #get the associated link functions
    vertices  = G['V'] #get the list of vertices
    edges = G['A'] #get the list of edges
    Y = G['Y']
    E = graph[2] #get the expression we want to evaluate
    X = {}
    Xs =[] #set of samples

    #First we topologically sort the vertices
    vertices = topologicalSort(vertices,edges)
    #initialize the child-parent mapping with empty lists:
    for i in vertices:
        P[i] = []
        #get the map Q of unobserved variables
        if i.find('sample') == 0:
        	X[i] = i
        	Q[i] = linkFuncs[i][1]
    P = childParentMapping(vertices,edges,P)

    #set X_0, by sampling every node
    for i in X:
        linkFuncsEval = bindingVars(X,linkFuncs[i])
        X[i]  = deterministic_eval(linkFuncsEval)

    for s in range(num_samples):
        Xret = gibbs_step(X)

        if type(E) == str:
        	#there is an assumption that if the return expression E is singular
    		#then it belongs to some node
        	retExp = Xret[E]
        else:
        	# find and replace the variables with the results from evaluating the link functions
        	retExp = bindingVars(Xret,E)

        Xs.append(deterministic_eval(retExp))
        X = Xret.copy()

    return Xs

def getMarkovBlanket(x):
	nodes = [x]
	parentsX = P[x]
	for i in parentsX:
		if not(i in nodes) and not(i == x):
			nodes.append(i)
	childrenX = edges[x]
	for i in childrenX:
		if not(i in nodes) and not(i == x):
			nodes.append(i)
	for i in childrenX:
		parentChild = P[i]
		for j in parentChild:
			if not(j in nodes) and not(j == x):
				nodes.append(j)
	return nodes

	nodes.append(P[x]) #get the parents of node x
	nodes.append(edges[x]) #get the children of node x
	#get the parents of the children without including node x
	for i in edges[x]:
		if not(P[i] in nodes) and not(P[i] == x):
			nodes.append(P[i])
	return nodes	