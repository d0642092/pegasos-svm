import numpy as np
from random import randint

class Pegasos(): 
    def __init__(self,x,y,iteration=2000,lam=1,epsilon=0):
        self.x = x
        self.y = y
        self.num_S = len(y)
        self.iteration = iteration
        self.lam = lam
        self.epsilon = epsilon

    def pegasos(self, weights=None):
        if type(weights) == type(None): weights = np.zeros(self.x[0].shape)
        for i in range(self.iteration):
            it = randint(0, self.num_S-1)
            step = 1/(self.lam*(i+1))
            decision = self.y[it] * weights @ self.x[it].T
            if decision < 1:
                weights = (1 - step*self.lam) * weights + step*self.y[it]*self.x[it]
            else:
                weights = (1 - step*self.lam) * weights
        return weights


    def kernel_function(self, x, y):
        mean = np.linalg.norm(x - y)**2
        variance = 0.02
        ans = np.exp(-mean*(variance))
        return ans

    def kernelized(self):
        np.random.seed(0) #2000
        itsets = np.random.randint(self.num_S, size=self.iteration)
        alpha = np.zeros(self.num_S,dtype=np.int8)
        for iter, item in enumerate(itsets):
            decision = 0
            for j in range(self.num_S):
                if alpha[j]!=0:
                    decision += alpha[j] * self.y[j] * self.kernel_function(self.x[item], self.x[j])
            decision = (self.y[item]*decision)/(self.lam*(iter+1))
            if decision < 1-self.epsilon:
                alpha[item] += 1
            else:
                alpha[item] = alpha[item]
            print("Index: {index:>4d}, Select: {item:>5d}, weight: {weights:>d}, decision: {decisions: .2f}".format(index=iter,item=item,weights=alpha[item],decisions=decision))
        print("train done")
        
        return alpha




            
            
