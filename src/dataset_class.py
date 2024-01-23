from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt

class Dataset():

    def __init__(self, data_dir, target, labels_to_load=[0,1]):
        self.labels_to_load = labels_to_load
        self.target = target
        self.mnist_loader = MNIST(data_dir)
        print('Loading dataset...')

        self.xtrain, self.ytrain = self.mnist_loader.load_training()
        self.xtrain = np.array(self.xtrain, dtype=np.float64)
        self.ytrain = np.array(self.ytrain, dtype=np.float64)
        print('xtrain')
        self.xtrain, self.ytrain,self.num_ytrain = self.trim_dataset(self.xtrain, self.ytrain)

        self.xtest, self.ytest = self.mnist_loader.load_testing()
        self.xtest = np.array(self.xtest, dtype=np.float64)
        self.ytest = np.array(self.ytest, dtype=np.float64)
        print('xtest')
        self.xtest, self.ytest,self.num_ytest = self.trim_dataset(self.xtest,self.ytest)
        print('Dataset loaded')

    def trim_dataset(self, x, y):
        xdata, ydata, truelabel = [],[],[]
        isOthers = False
        targetNum = 0
        if self.target[1] == "others":
            isOthers = True
        for i in range(len(y)):
            tmp=np.clip(x[i],0,1)
            if y[i] == self.target[0]:
                targetNum +=1
                ydata.append(1)
                xdata.append(tmp)
                truelabel.append(y[i])
            elif isOthers:
                ydata.append(-1)
                xdata.append(tmp)
                truelabel.append(y[i])
            elif y[i] == self.target[1]:
                ydata.append(-1)
                xdata.append(tmp)
                truelabel.append(y[i])
        print("num: ",targetNum)
        return np.array(xdata,dtype=np.int8), np.array(ydata,dtype=np.int8), np.array(truelabel,dtype=np.int8)

    def checkimg(self, data, label, position):
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