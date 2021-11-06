import torch

from primitivesAD import deterministic_eval,topologicalSort,childParentMapping,bindingVars
import distributions as dist
import numpy as np

G = []
linkFuncs = []
vertices = []
edges = []
sig = {}
Y = []
P = {}

def bbvi(graph,T,num_samples,lr):
    "This function does Metropolis-Hastings within Gibbs"
    #set the global context
    global G
    global linkFuncs
    global vertices
    global edges
    global Y 
    global P
    global sig

    G = graph[1] #get the graph structure
    linkFuncs = G['P'] #get the associated link functions
    vertices  = G['V'] #get the list of vertices
    edges = G['A'] #get the list of edges
    Y = G['Y']
    E = graph[2] #get the expression we want to evaluate
    R = {}
    X = {} #set of random variables
    jll = [] #set of joint-log-lik for each sample
    gradients = []
    logWeights = torch.zeros(T,num_samples)
    #First we topologically sort the vertices
    vertices = topologicalSort(vertices,edges)
    #initialize the child-parent mapping with empty lists:
    for i in vertices:
        P[i] = []
        if i.find('sample') == 0:
            X[i] = i
    P = childParentMapping(vertices,edges,P)
    sig['logW'] = 0
    sig['G'] = {}
    sig['Q'] = {}
    sig['Qp'] ={}
    sig['opt'] = {}
    ret = []
    #set R_0, by sampling every node from the prior and set the initial map of G and Q to empty,
    #they get populated in deterministic_eval
    for v in vertices:
        linkFuncsEval = bindingVars(R,linkFuncs[v])
        R[v]  = deterministic_eval(linkFuncsEval,sig,v)

    for v in sig['Q']:
        sig['opt'][v] = [torch.optim.Adam(sig['Q'][v].Parameters(),lr),sig['Q'][v]]
        sig['Qp'][v] = [torch.stack(sig['Q'][v].Parameters()).detach()]

    for t in range(T):
        gradients.append([])
        ret.append([])
        for s in range(num_samples):
            sig['G'] = {}
            sig['logW'] = 0
            for v in vertices:
            #    mb = getMarkovBlanket(v)
            #    for x in mb:
                linkFuncsEval = bindingVars(R,linkFuncs[v])
                R[v] = deterministic_eval(linkFuncsEval,sig,v)
            gradients[t].append(sig['G'].copy())
            logWeights[t,s] = sig['logW'].detach()

            if type(E) == str:
                retExp = R[E]
            else:
                retExp = bindingVars(R,E)
            ret[t].append(deterministic_eval(retExp,sig,[]))
        ghat = elbo_grad(gradients[t],logWeights[t,:])
        print(t)
        print(logWeights[t,:].mean())
        optimize(sig['Q'],ghat,sig['opt'],sig['Qp'],t)
       # input('before after')
        
    return ret,logWeights,sig['Q'],sig['Qp']

def getMarkovBlanket(x):
	nodes = [x]
	parentsX = P[x]
	for i in parentsX:
		if not(i in nodes) and not(i == x):
			nodes.append(i)
	if x in edges.keys():
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


def elbo_grad(G,logW):
    #merging list of dictionaries: 
    # https://stackoverflow.com/questions/3494906/how-do-i-merge-a-list-of-dicts-into-a-single-dict
    # this line is horribly unreadable, but:
    # for every key 'k' in each dictionary 'd' of list 'G'
    # insert its value 'v'.
    # note that the values in this merged gradient object are irrelevant
    # we just want to know all the keys that exists in all L gradient samples
    # because different traces may have invoked different gradient addresses (or nodes)
    Gmerg = {k: v for d in G for k, v in d.items()}
  #  return Gmerg
    #initialize my F dictionary 
    #with the correct number of keys and empty lists
    F = {k: [] for k in Gmerg}
    Gs = {k: [] for k in Gmerg} 
    L = len(logW)
    ghat = {}
    for v in Gmerg:
        sizeG = torch.stack(Gmerg[v]).shape
        for s in range(L):
            if v in G[s].keys():
                F[v].append(torch.stack(G[s][v])*logW[s])
            else:
                F[v].append(torch.zeros(sizeG))
                G[s][v].append(torch.zeros(sizeG))
        #this picks out all gradients from 1:L of G in a list for variable v
        GL = torch.stack([torch.stack(g[v]) for g in G])
        Fv = torch.stack(F[v])

        bhat = computeBaseline(Fv,GL)

        ghat[v] = torch.sum((Fv - bhat*GL)/L,0)

    return ghat

def computeBaseline(Fv,GL):
    C = []
    #Check to see if the elements of Fv are multi-dimensional
    if len(Fv[0].shape) > 1:
        n,d = Fv[0].shape
        for i in range(n):
            C_i = []
            #collect the cov(Fi,Gi) terms
            for j in range(d):
                C_ij = np.cov(Fv.detach().numpy()[:,i,j],GL.numpy()[:,i,j], rowvar = True)
                C_i.append(C_ij[0,1])
        C.append(C_i)
        Cov = torch.tensor(C)
        #need to get ride of NaN, sum all the cov(Fi,Gi) terms
        bhat = torch.nan_to_num(Cov.sum()/GL.var(0).sum())
    else:
        for i in range(Fv[0].shape[0]):
            C_i = np.cov(Fv.detach().numpy()[:,i],GL.numpy()[:,i], rowvar = True)
            C.append(C_i[0,1])
        Cov = torch.tensor(C)
        #again need to get ride of NaN
        bhat = torch.nan_to_num(Cov.sum()/GL.var(0))

    # for some reason numpy turns the above into float64, everything else (and default torch floating point numbers) is in torch.float32
    # so truncate back to torch.float32
    return torch.tensor(bhat,dtype = torch.float32)



def optimize(Q,ghat,optimizers,Qp,t):
    for v in ghat:
        [opt,Dobj] = optimizers[v]
        grad = ghat[v]
        i = 0
        for p in Dobj.Parameters():
            p.grad = -torch.nan_to_num(grad[i].detach())
            i+=1
        opt.step()
        opt.zero_grad()
        
        i = 0
        for p in Q[v].Parameters():
            p.data = Dobj.Parameters()[i] #+ 1e-8/(t+1)*grad[i]
            i += 1
        Qp[v].append(torch.stack(Q[v].Parameters()).detach())

       # i = 0
       # for p in Q[v].Parameters():
       #     p.data = Dobj.Parameters()[i].detach()
       #     i += 1 


