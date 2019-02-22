import sys
import random
#import matplotlib.pyplot as pyplot

def shell():
    if len(sys.argv) <= 1:
        filename = 'housing.data'
    else:
        filename = sys.argv[2]
    data = readData(filename)

    print('x:', len(data[0]), '*', len(data[0][0]))
    print('y:', len(data[1]))

    x=data[0]
    y=data[1]

    normalize(x)
    shuffle(x, y)

    trainN=int(len(x)*0.8)
    trainx=x[0:trainN]
    trainy=y[0:trainN]
    testx=x[trainN:]
    testy=y[trainN:]


    for i in range(1):
        print('run',i)
        itc=100000
        while True:
            w = LR_learn(trainx, trainy, 101, itc, 5e-5)
            trainErr=meanSquaredError(LRs_predict(w, trainx), trainy)
            testErr=meanSquaredError(LRs_predict(w, testx), testy)
            print('mean squared error on train set:', trainErr)
            print('mean squared error on test set:', testErr)
            if trainErr<50:
                break
            else:
                itc+=100000

def meanSquaredError(v1,v2):#求两数组均方误差
    l=min(len(v1),len(v2))
    ret=0
    for i in range(l):
        d=v1[i]-v2[i]
        ret+=d*d/l
    return ret

def LRs_predict(w,x):#x矩阵乘列向量w 返回列向量y
    y=[]
    for xi in x:
       y.append(innerProduct(w,xi))
    return y

def LR_learn(x, y, batchSize, iterationCount, learningRate):#输入训练集x和y，训练w
    print('\ntrain sample size:',len(y),'\nbatch size:',batchSize,
          '\niteration count:',iterationCount,'\nlearning rate:',learningRate)
    w = []
    for i in range(len(x[0])):
        w.append(random.random())

    for it in range(iterationCount):
        samplei = 0
        while samplei < len(y):
            for factori in range(len(x[0])):
                errorSum = 0
                batchMaxi = min(batchSize + samplei, len(y))
                size = batchMaxi - samplei
                while samplei < batchMaxi:
                    error = innerProduct(w, x[samplei]) - y[samplei]
                    errorSum += error * x[samplei][factori] / size
                    samplei += 1

                w[factori] -= learningRate * errorSum
    print('w:',w)
    return w


def innerProduct(w, x):#向量内积
    l = min(len(w), len(x))
    ret = 0
    for i in range(l):
        ret += w[i] * x[i]
    return ret


def shuffle(x, y):#混洗x矩阵伴随y列向量（交换行）
    for i in range(len(y)):
        tx = x[i].copy()
        ty = y[i]
        j = random.randint(0, len(y) - 1)
        x[i] = x[j].copy()
        y[i] = y[j]
        x[j] = tx.copy()
        y[j] = ty


def normalize(x):#归一化x的各个列
    maxx = x[0].copy()
    minx = x[0].copy()
    for xi in x:
        for i in range(len(xi)):
            if xi[i] > maxx[i]:
                maxx[i] = xi[i]
            if xi[i] < minx[i]:
                minx[i] = xi[i]

    scale = []
    for i in range(len(maxx)):
        scale.append(maxx[i] - minx[i])

    for xi in x:
        for i in range(len(xi)):
            if scale[i] != 0:
                xi[i] = (xi[i] - minx[i]) / scale[i]


def readData(filename):#读数据，返回x矩阵和y列向量，数据前面所有列组成x，最后一列是y
    f = open(filename, 'r')
    retx = []
    rety = []
    for line in f:
        array = line.split(' ')
        xi = []
        for i in array[0:-1]:
            if len(i) > 0:
                xi.append(float(i))
        retx.append(xi)
        rety.append(float(array[-1]))
    return (retx, rety)


if __name__ == '__main__':
    shell()
