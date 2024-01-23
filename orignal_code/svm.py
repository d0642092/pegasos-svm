#!/usr/bin/env python

import os
import sys
import argparse
from tabnanny import check
from turtle import pos
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
from pegasos import *
import time
class Dataset():

    def __init__(self, data_dir, labels_to_load=[0,1]):
        self.labels_to_load = labels_to_load
        self.mnist_loader = MNIST(data_dir)
        print('Loading dataset...')

        self.xtrain, self.ytrain = self.mnist_loader.load_training()
        self.xtrain = np.array(self.xtrain, dtype=np.float64)
        self.ytrain = np.array(self.ytrain, dtype=np.float64)
        print('xtrain')
        self.xtrain, self.ytrain,self.num_ytrain = self.trim_dataset(self.xtrain, self.ytrain)

        self.xtest, self.ytest = self.mnist_loader.load_testing()
        self.xtest = np.array(self.xtest, dtype=np.float64)
        # self.ytest = np.array(self.ytest, dtype=np.float64)
        # print(self.ytest)
        self.ytest = np.array(self.ytest, dtype=np.float64)
        print('xtest')
        self.xtest, self.ytest,self.num_ytest = self.trim_dataset(self.xtest,self.ytest)
        print('Dataset loaded')

    def trim_dataset(self, x, y):
        xtrain = []
        ytrain = []
        mapNum = []
        k = 0
        for i in range(len(y)):
            a=np.clip(x[i],0,1)
            
            if y[i] == 8:
                k+=1
                ytrain.append(1)
                xtrain.append(a)
                mapNum.append(y[i])
            # elif y[i] == 1:
            #     ytrain.append(-1)
            #     xtrain.append(a)
            #     mapNum.append(y[i])
            else:
                ytrain.append(-1)
                mapNum.append(y[i])
                xtrain.append(a)
        print("num: ",k)
        return np.array(xtrain), np.array(ytrain), np.array(mapNum)

def kernel_function(x, y):
    mean = np.linalg.norm(x - y)**2
    variance = 1
    return np.exp(-mean/(2*variance))

def paper_kernel_function(x, y):
    mean = np.linalg.norm(x - y)**2
    variance = 0.02
    ans = np.exp(-mean*(variance))
    return ans

def parse_arguments():
    # args
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--kernel', default=False, action='store_true')
    parser.add_argument('--lambda', default=1, type=float)
    return parser.parse_args()

def kernelized_svm(args, data):
    weights = kernelized_pegasos(
            x=data.xtrain,
            y=data.ytrain,
            kernel=kernel_function,
            iterations=args.iterations
    )
    errors = 0
    for i in range(len(data.ytest)):
        decision = 0
        print(i)
        for j in range(len(data.ytrain)):
            if weights[j] != 0:
                decision += weights[j]*data.ytrain[j]*kernel_function(data.xtrain[j], data.xtest[i])
        if decision < 0:
            prediction = -1
        else:
            prediction = 1
        if prediction != data.ytest[i]: errors += 1
    return 1 - errors/len(data.ytest)

def kernelized_svm_paper(args, data):
    startTrain = time.time()
    weights = kernelized(
            x=data.xtrain,
            y=data.ytrain,
            kernel=paper_kernel_function,
            iterations=args.iterations
    )
    endTrain = time.time() #startTest
    errors = 0
    for i in range(len(data.ytest)):
        # i = randint(0,10000)
        decision = 0
        
        for j in range(len(data.ytrain)):
            if weights[j] != 0:
                decision += weights[j]*data.ytrain[j]*paper_kernel_function(data.xtrain[j], data.xtest[i])
        if decision < 0:
            prediction = -1
            # print("Nega Index:{index:<4d}, label:{item: d}, real_num:{num:<d}, decision:{decisions: .2f}".format(index=i,item=data.ytest[i],num=data.num_ytest[i],decisions=decision))
        else:
            prediction = 1
            # print("Posi Index:{index:<4d}, label:{item: d}, real_num:{num:<d}, decision:{decisions: .2f}".format(index=i,item=data.ytest[i],num=data.num_ytest[i],decisions=decision))

        if prediction != data.ytest[i]: 
            print("Error Index: {index:>4d}, label: {item: d}, real_num: {num: d}, decision: {decisions: .2f}".format(index=i,item=int(data.ytest[i]),num=int(data.num_ytest[i]),decisions=decision))
            # checkimg(data.xtest,data.ytest, i)
            errors += 1
    print("Errors: ", errors)
    endTest = time.time()
    print("Training time: {: .2f}, Test time: {: .2f}, Total time: {: .2f}".format(endTrain-startTrain, endTest-endTrain, endTest-startTrain))
    return 1 - errors/len(data.ytest)

def svm(args, data):
    weights = pegasos(
            x=data.xtrain,
            y=data.ytrain,
            iterations=args.iterations
    )
    errors = 0
    for i in range(len(data.ytest)):
        decision = weights @ data.xtest[i].T
        if decision < 0:
            prediction = -1
        else:
            prediction = 1
        if prediction != data.ytest[i]: errors += 1
    return 1 - errors/len(data.ytest)

def checkimg(data,label,position):
    for i in position:
        print(i)
        curr_img   = np.reshape(data[i, :], (28, 28))
        #
        plt.title(label[i])
        plt.axis('off')
        plt.imshow(curr_img,cmap="gray")
        # plt.show()
        plt.show(block=False)
        plt.pause(3) # 3 seconds, I use 1 usually
        fig = plt.gcf()
        plt.close(fig)

def main():
    args = parse_arguments()
    data = Dataset(args.dataset_dir)

    if args.kernel:
        print('Using RBF kernel')
        # checkimg(data.xtrain,data.ytrain,[15920, 38870, 2001,3273,16730])
        # accuracy = kernelized_svm(args, data)
        accuracy = kernelized_svm_paper(args, data)
        print("svm 300000 epsilion done")
    else:
        accuracy = svm(args, data)
    print('Accuracy:', accuracy)

main()
