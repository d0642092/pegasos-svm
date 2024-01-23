import gc
import time
import argparse
import os

from dataset_class import Dataset
from pegasos_class import Pegasos


def parse_arguments():
    # args
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--kernel', default=False, action='store_true')
    parser.add_argument('--lambda', default=1, type=float)
    return parser.parse_args()
    
def evaluate_svm(testNum,weights,trainType=None):
    errors = 0
    if trainType == "kernelized":
        weightDict = {}
        for index, value in enumerate(weights):
            if value != 0:
                weightDict[index] = value*data.ytrain[index]
        print("Need calculate: ",len(weightDict))
        # os.system("pause")
        weights = weightDict.copy()
        del weightDict
        gc.collect()
        
    for i in range(testNum):
        decision = calDecision(trainType, weights, i)
        if decision < 0:
            prediction = -1
        else:
            prediction = 1
        if prediction != data.ytest[i]: 
            print("Error Index: {index:>4d}, label: {item: d}, real_num: {num: d}, decision: {decisions: .2f}".format(index=i,item=int(data.ytest[i]),num=int(data.num_ytest[i]),decisions=decision))
            # data.checkimg(data=data.xtest,label=data.ytest,position=i)  
            errors += 1
    print("Errors: ", errors)
    return 1 - errors/testNum

def calDecision(trainType, weights, i):
    decision = 0
    if trainType == "kernelized":
        for index, w in weights.items():
            decision += w * pegasos.kernel_function(data.xtrain[index], data.xtest[i])
    else:
        decision = weights @ data.xtest[i].T
    return decision

if __name__ == "__main__":
    args = parse_arguments()
    data = Dataset(args.dataset_dir,[8,"others"]) # "others"除選擇的數字以外(可改成指定數字ex:1)
    trainType="kernelized"
    startTrain = time.time()
    # epsilon=0.00144
    pegasos = Pegasos(x=data.xtrain, 
                      y=data.ytrain,
                      iteration=args.iterations,
                      lam=1.67*10**-5,
                      epsilon=0)
    if args.kernel:
        print('Using RBF kernel')
        # data.checkimg(data=data.xtrain,label=data.ytrain,position=[15920, 38870, 2001,3273,16730])  
        weight = pegasos.kernelized()
        print("kernelized svm done")
    else:
        trainType="Binary"
        weight = pegasos.pegasos()
        print("svm done")
    
    endTrain = time.time() #startTest
    accuracy = evaluate_svm(testNum=len(data.ytest),weights=weight,trainType=trainType)
    endTest = time.time()
    print(trainType+" svm done")
    print("Training time: {: .2f}, Test time: {: .2f}, Total time: {: .2f}".format(endTrain-startTrain, endTest-endTrain, endTest-startTrain))
    print('Accuracy:', accuracy)
