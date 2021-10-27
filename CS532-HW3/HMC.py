import torch

from primitivesAD import deterministic_eval,topologicalSort,childParentMapping,bindingVars
import torch.distributions as dist

G = []
linkFuncs = []
vertices = []
edges = []
Y = []
P = {}
Q = {}

def grad(X):
	gradU = {}
	logprob = 0
	for x in X:
		evalX = bindingVars(X,linkFuncs[x])
		if(evalX[0] == 'sample*'):
			evalX[0] = 'observe*'
			evalX.append(X[x].clone().detach().requires_grad_(True))
		#observe on each X
		
		u = deterministic_eval(evalX)
		logprob -= u[0]
		gradU[x] = u[1]

	logprob.backward()
	for x in gradU:
		gradU[x] = gradU[x].grad

	return gradU

def leapfrog(X,R,T,eps):
	Rh = {}
	gradU = grad(X)
	for x in X:
		Rh[x] = R[x] - 0.5*eps*gradU[x]
	Xt = X.copy()
	for t in range(1,T-1):
		for x in X:
			Xt[x] = Xt[x] + eps*Rh[x]
		gradU = grad(Xt)
		for x in X:
			Rh[x] = Rh[x] - eps*gradU[x]
	XT ={}
	RT = {}
	for x in X:
		XT[x] = Xt[x] + eps*Rh[x]
	gradU = grad(XT)
	for x in X:
		RT[x] = Rh[x] - 0.5*eps*gradU[x]
	return XT,RT


def HMC(graph,num_samples,T,eps):
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

    for i in vertices:
        #get the unobserved variables and leaves them as variables
        if i.find('sample') == 0:
        	X[i] = i

    #set X_0, by sampling every unobserved node
    for i in X:
        linkFuncsEval = bindingVars(X,linkFuncs[i])
        X[i]  = deterministic_eval(linkFuncsEval)

    # add to X the observed Y's
    for i in Y:
    	X[i] = torch.tensor(float(Y[i]),requires_grad = True)
    M = 1
    R = {}

    for s in range(num_samples):
    	for x in X:
    		R[x] = dist.Normal(0,M).sample()

    	[Xp,Rp] = leapfrog(X,R,T,eps)
    	u = dist.Uniform(0,1).sample()

    	if u < torch.exp(-H(Xp,Rp,M) + H(X,R,M)):
    		X = Xp.copy()
    	
    	if type(E) == str:
        	#there is an assumption that if the return expression E is singular
    		#then it belongs to some node
    		retExp = X[E]
    	else:
    		# find and replace the variables with the results from evaluating the link functions
    		retExp = bindingVars(X,E)

    	Xs.append(deterministic_eval(retExp))
        	
        
    return Xs

def H(X,R,M):
	U = 0
	Rv = []
	for x in X:
		evalX = bindingVars(X,linkFuncs[x])
		if(evalX[0] == 'sample*'):
			evalX[0] = 'observe*'
			evalX.append(X[x])
		U-=deterministic_eval(evalX)[1]
		Rv.append(R[x])
	Rv = torch.stack(Rv)
	M = torch.eye(len(Rv))*M
	K = 0.5*torch.matmul(Rv,torch.matmul(torch.inverse(M),Rv))

	return U + K	