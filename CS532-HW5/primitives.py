import torch
import torch.distributions as dist



class Normal(dist.Normal):
    
    def __init__(self, alpha, loc, scale):
        
        if scale > 20.:
            self.optim_scale = scale.clone().detach().requires_grad_()
        else:
            self.optim_scale = torch.log(torch.exp(scale) - 1).clone().detach().requires_grad_()
        
        
        super().__init__(loc, torch.nn.functional.softplus(self.optim_scale))
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.loc, self.optim_scale]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]
         
        return Normal(*ps)
    
    def log_prob(self, x):
        
        self.scale = torch.nn.functional.softplus(self.optim_scale)
        
        return super().log_prob(x)

class Bernoulli(dist.Bernoulli):
    
    def __init__(self,alpha, probs=None, logits=None):
        if logits is None and probs is None:
            raise ValueError('set probs or logits')
        elif logits is None:
            if type(probs) is float:
                probs = torch.tensor(probs)
            logits = torch.log(probs/(1-probs)) ##will fail if probs = 0
        #
        super().__init__(logits = logits)
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.logits]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        logits = [p.clone().detach().requires_grad_() for p in self.Parameters()][0]

        return Bernoulli(logits = logits)
    
class Categorical(dist.Categorical):
    
    def __init__(self,alpha, probs=None, logits=None, validate_args=None):
        
        if (probs is None) == (logits is None):
            raise ValueError("Either `probs` or `logits` must be specified, but not both.")
        if probs is not None:
            if probs.dim() < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            probs = probs / probs.sum(-1, keepdim=True)
            logits = dist.utils.probs_to_logits(probs)
        else:
            if logits.dim() < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        super().__init__(logits = logits)
        self.logits = logits.clone().detach().requires_grad_()
        self._param = self.logits

    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.logits]

    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        logits = [p.clone().detach().requires_grad_() for p in self.Parameters()][0]
        
        return Categorical(logits = logits)    

class Dirichlet(dist.Dirichlet):
    
    def __init__(self, alpha,concentration):
        #NOTE: logits automatically get added
        super().__init__(concentration)
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.concentration]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        concentration = [p.clone().detach().requires_grad_() for p in self.Parameters()][0]
        
        return Dirichlet(concentration)

class Gamma(dist.Gamma):
    
    def __init__(self, alpha,concentration, rate, copy=False):
        if rate > 20. or copy:
            self.optim_rate = rate.clone().detach().requires_grad_()
        else:
            self.optim_rate = torch.log(torch.exp(rate) - 1).clone().detach().requires_grad_()
        
        
        super().__init__(concentration, torch.nn.functional.softplus(self.optim_rate))
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.concentration, self.optim_rate]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        concentration,rate = [p.clone().detach().requires_grad_() for p in self.Parameters()]
        return Gamma(concentration, rate, copy = True)

    def log_prob(self, x):
        
        self.rate = torch.nn.functional.softplus(self.optim_rate)
        
        return super().log_prob(x)

class Uniform:

    def __init__(self,alpha,low,high,copy=False):
        if low >= high:
            lowNew = high.clone().detach().requires_grad_()
            highNew = low.clone().detach().requires_grad_()
            low = lowNew
            high = highNew

        self.low = low
        self.high = high
    def Parameters(self):
        return [self.low,self.high]

    def make_copy_with_grads(self):
        low,high = [p.clone().detach().requires_grad_() for p in self.Parameters()]
        return Uniform(low,high)

    def sample(self):
        return dist.Uniform(self.low,self.high).sample().nan_to_num()

    def log_prob(self,x):
        s = 1
        return torch.log(0.5/(self.high - self.low)*(-torch.tanh(s*(x-self.high))+torch.tanh(s*(x+self.low))))

        

def push_addr(alpha, value):
    return alpha + value

def Vector(*args):
    addr = args[0]
    if len(args[1:]) == 0:
        return []
    return torch.stack([i for i in args[1:]])

def Get(*args):

    addr,d,ind = args
    addr = args[0]
    d = args[1]
    ind = args[2]
    if type(ind) == type(torch.tensor(1.)):
        ind = int(ind)
    return d[ind]

def Put(*args):
    addr,d,ind,val = args
    if type(d) == dict:
        dRet = d.copy()
    else:
        dRet = d.clone()
    if type(ind) == type(torch.tensor(1.)):
        ind = int(ind)
    dRet[ind] = val    
    return dRet

def First(*args):
    addr = args[0]
    vec = args[1]
    return vec[0]

def Second(*args):
    addr = args[0]
    vec = args[1]
    return vec[1]

def Last(*args):
    addr = args[0]
    vec = args[1]
    return vec[len(vec)-1]

def Rest(*args):
    addr = args[0]
    vec = args[1]
    return vec[1:]

def Append(*args):
    addr = args[0]
    vec = args[1]
    val = args[2]
    return torch.cat((vec,torch.tensor([val])),0)

def dirac(*args):
    addr = args[0]
    sigma = 0.1
    return Normal(d,sigma)

def HashMap(*args):
    addr = args[0]
    kvp = args[1:]
    retD = {}
    i = 0
    while(i < len(kvp)-1):
        if type(kvp[i]) == type(torch.tensor(1.)):
            retD[int(kvp[i])] = kvp[i+1]
        else:
            retD[kvp[i]] =  kvp[i+1]
        i += 2
    return retD

def Cons(*args):
    addr,l,val = args
    return torch.cat((torch.tensor([val]),torch.tensor(l)),0)

def Conj(*args):
    addr,l1,val = args
    return torch.cat((l1,torch.tensor([val])),0)

env = { 'sqrt': lambda _,a: torch.sqrt(a),
        '+': lambda _,a,b: torch.add(a,b),
        '-': lambda _,a,b: torch.sub(a,b),
        '/': lambda _,a,b: torch.div(a,b),
        '*': lambda _,a,b: torch.mul(a,b),
        'exp': lambda _,a: torch.exp(a),
        'abs': lambda _,a: torch.abs(a),
        'log': lambda _,a: torch.log(a),
        '>': lambda _,a,b: torch.greater(a,b),
        '<': lambda _,a,b: torch.less(a,b),
        '=': lambda _,a,b: torch.equal(a,b),
        'or': lambda _,a,b: a or b,
        'and': lambda _,a,b: a and b,
        'empty?': lambda _,a: len(a)==0,
        'true': torch.tensor(1.0),
        'false': torch.tensor(0.0),
        'normal': Normal,
        'uniform': Uniform,
        'uniform-continuous': Uniform,
        'exponential': lambda _,x: dist.Exponential(x),
        'beta': lambda _,a,b: dist.Beta(a,b),
        'discrete': Categorical,
        'dirichlet': Dirichlet,
        'flip': Bernoulli,
        'gamma': Gamma,
        'dirac': None,
        'vector': Vector,
        'mat-transpose': lambda _, M: M.t(),
        'mat-add': lambda _,a,b: torch.add(a,b),
        'mat-mul': lambda _,a,b: torch.matmul(a,b),
        'mat-repmat': lambda _,M,a,b: M.repeat(int(a),int(b)),
        'mat-tanh': lambda _,a: torch.tanh(a),
        'get': Get,
        'put': Put,
        'first': First,
        'second': Second,
        'rest': Rest,
        'last': Last,
        'append': Append,
        'hash-map': HashMap,
        'cons': Cons,
        'conj': Conj,
        'peek': lambda _,d: d[-1],
        'push-address': push_addr}







