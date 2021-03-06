import numpy as np
##Q3


##first define the probability distributions as defined in the excercise:

#define 0 as false, 1 as true
def p_C(c):
    p = np.array([0.5,0.5])
    
    return p[c]


def p_S_given_C(s,c):
    p = np.array([[0.5,0.9],[0.5,0.1]])
    return p[s,c]
    
def p_R_given_C(r,c):
    p = np.array([[0.8,0.2],[0.2,0.8]])
    return p[r,c]

def p_W_given_S_R(w,s,r):
    
    p = np.array([
            [[1.0,0.1],[0.1,0.01]],   #w = False
            [[0.0,0.9],[0.9,0.99]],   #w = True
            ])
    return p[w,s,r]


##1. enumeration and conditioning:
    
## compute joint:
p = np.zeros((2,2,2,2)) #c,s,r,w
for c in range(2):
    for s in range(2):
        for r in range(2):
            for w in range(2):
                p[c,s,r,w] = p_C(c)*p_S_given_C(s,c)*p_R_given_C(r,c)*p_W_given_S_R(w,s,r)
                
## condition and marginalize:
p_W = np.array([p[:,:,:,0].sum(),p[:,:,:,1].sum()])
p_C_given_W = np.array([p[0,:,:,1].sum()/p_W[1],p[1,:,:,1].sum()/p_W[1]])

print('There is a {:.2f}% chance it is cloudy given the grass is wet'.format(p_C_given_W[1]*100))



##2. ancestral sampling and rejection:
num_samples = 10000
samples = np.zeros(num_samples)
rejections = 0
i = 0
j = 0
while i < num_samples:
    [c_s,s_s,r_s,w_s] = np.random.uniform(0,1,4)
    c = 1 if c_s < p_C(1) else 0
    s = 1 if s_s < p_S_given_C(1,c) else 0
    r = 1 if r_s < p_R_given_C(1,c) else 0
    w = 1 if w_s < p_W_given_S_R(1,s,r) else 0

    if(w == 0):
        rejections += 1
    else:
        samples[j] = c
        j += 1
    i += 1

# we want to know the probability of cloudy on the fraction of accepted samples
# so we remove the samples that are rejected.
# Only keep the j accepted samples, 
samples = samples[0:j]

print('The chance of it being cloudy given the grass is wet is {:.2f}%'.format(samples.mean()*100))
print('{:.2f}% of the total samples were rejected'.format(100*rejections/(samples.shape[0]+rejections)))


#3: Gibbs
# we can use the joint above to condition on the variables, to create the needed
# conditional distributions:


#we can calculate p(R|C,S,W) and p(S|C,R,W) from the joint, dividing by the right marginal distribution
#indexing is [c,s,r,w]
p_R_given_C_S_W = p/p.sum(axis=2, keepdims=True)
p_S_given_C_R_W = p/p.sum(axis=1, keepdims=True)


# but for C given R,S,W, there is a 0 in the joint (0/0), arising from p(W|S,R)
# but since p(W|S,R) does not depend on C, we can factor it out:
#p(C | R, S) = p(R,S,C)/(int_C (p(R,S,C)))

#first create p(R,S,C):
p_C_S_R = np.zeros((2,2,2)) #c,s,r
for c in range(2):
    for s in range(2):
        for r in range(2):
            p_C_S_R[c,s,r] = p_C(c)*p_S_given_C(s,c)*p_R_given_C(r,c)
            
#then create the conditional distribution:
p_C_given_S_R = p_C_S_R[:,:,:]/p_C_S_R[:,:,:].sum(axis=(0),keepdims=True)



##gibbs sampling
num_samples = 10000
samples = np.zeros(num_samples)
state = np.zeros(4,dtype='int') # use this as the initial state
#c,s,r,w, set w = True
state[3] = 1
i = 0
while i < num_samples:
    [c_s,s_s,r_s] = np.random.uniform(0,1,3)
    #c ~ p(C|S,R)
    state[0] = 1 if c_s < p_C_given_S_R[1,state[1],state[2]] else 0
    #s ~ p(S|C,R,W) 
    state[1] = 1 if s_s < p_S_given_C_R_W[state[0],1,state[2],state[3]] else 0
    #r ~ P(R|C,S,W)
    state[2] = 1 if r_s < p_R_given_C_S_W[state[0],state[1],1,state[3]] else 0
    # no update on sate[3] = w, since we are given that this is true
    samples[i] = state[0]
    i += 1

print('The chance of it being cloudy given the grass is wet is {:.2f}%'.format(samples.mean()*100))
