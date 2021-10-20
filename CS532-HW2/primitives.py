import torch

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
		#for now we treat this as sample and ignore observed values
		Dobj = args[0]
		ret = Dobj.sample()
	elif e == 'if':
		if args[0]:
			ret = args[1]
		else:
			ret = args[2]
	elif e == 'get':
		ind = int(args[1])
		ret = args[0][ind]
	elif e == 'mat-transpose':
		ret = op(args[0])
	elif e == 'mat-repmat':
		ret = args[0].repeat(int(args[1]),int(args[2]))
	elif e == 'mat-mul':
		ret = op(*args)
	else:
		ret = op(*args)

	return ret

def bindingVars(varDict,funcCode):
	evalFuncCode = funcCode.copy()
	for i in range(0,len(funcCode)):
		if type(evalFuncCode[i]) == list:
			evalFuncCode[i] = bindingVars(varDict,evalFuncCode[i])
		elif evalFuncCode[i] in varDict.keys():
			evalFuncCode[i] = float(varDict[evalFuncCode[i]]) #hacky for now
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