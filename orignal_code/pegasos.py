#!/usr/bin/env pytho

import enum
import math
import numpy as np
from random import randint

def pegasos(x, y, weights=None, iterations=2000, lam=1):
    if type(weights) == type(None): weights = np.zeros(x[0].shape)
    num_S = len(y)
    for i in range(iterations):
        it = randint(0, num_S-1)
        step = 1/(lam*(i+1))
        decision = y[it] * weights @ x[it].T
        if decision < 1:
            weights = (1 - step*lam) * weights + step*y[it]*x[it]
        else:
            weights = (1 - step*lam) * weights
        #weights = min(1, (1/math.sqrt(lam))/(np.linalg.norm(weights)))*weights
    return weights

def kernelized_pegasos(x, y, kernel, weights=None, iterations=2000, lam=1):
    num_S = len(y)
    if type(weights) == type(None): weights = np.zeros(num_S)
    for _ in range(iterations):
        it = randint(0, num_S)
        decision = 0
        for j in range(num_S):
            decision += weights[j] * y[it] * kernel(x[it], x[j])
        decision *= y[it]/lam
        if decision < 1:
            weights[it] += 1
    return weights

def kernelized(x, y, kernel, weights=None, iterations=2000, lam=1.67*10**-5):
    num_S = len(y)
    alpha = np.zeros(num_S,dtype=np.int8)
    eta = lam
    # epsilon = 0.00144
    np.random.seed(0)
    itsets = np.random.randint(num_S, size=iterations)
    # for _ in range(iterations):
    #     it = randint(0,num_S)
    for t, it in enumerate(itsets):
        decision = 0
        eta = lam*(t+1)
        for j in range(num_S):
            if alpha[j]!=0:
                decision += alpha[j] * y[j] * (kernel(x[it], x[j]))
        decision = (y[it]*decision)/eta
        if decision < 1:
            alpha[it] += 1
        else:
            alpha[it] = alpha[it]
        
        print("Index: {index:>4d}, Select: {item:>5d}, weight: {weights:>d}, decision: {decisions: .2f}".format(index=t,item=it,weights=alpha[it],decisions=decision))
    print("train done")
    
    return alpha
