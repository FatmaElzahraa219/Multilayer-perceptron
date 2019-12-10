import copy
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
from decimal import Decimal
f = open('C:\\Users\\User\\Desktop\\neuralTask1\\IrisData.txt', 'r')
def readdataset(feature1,feature2,class1,class2):
    f.readline()
    D = np.zeros((150, 4))
    lines = f.readlines()
    counter = 0
    temp = np.zeros((1, 5))
    for line in lines:
        temp = line.split(",")
        D[counter, 0:4] = temp[0:4]
        counter += 1
    return  D
