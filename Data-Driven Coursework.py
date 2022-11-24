from __future__ import print_function
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import itertools as IT

from random import seed
from random import randrange


def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values

def polydeg1(x,y):
    sum = 0
    newy = y.reshape((20,1))
    newx = x.reshape((20,1))
    ones = np.ones(newx.shape)
    newx = np.column_stack((ones,newx))
    A = np.linalg.inv(newx.T.dot(newx)).dot(newx.T).dot(newy)
    y_hat = A[0] + A[1]*x
    LSE = (y_hat - y)**2
    for i in range(len(LSE)):
        sum = sum + LSE[i]
    return A,sum


def polydeg3(x,y):
    sum = 0
    newy = y.reshape((20,1))
    newx = x.reshape((20,1))
    newx2 = x**2
    newx3 = x**3
    ones = np.ones(newx.shape)
    newx = np.column_stack((ones,newx,newx2,newx3))
    A = np.linalg.inv(newx.T.dot(newx)).dot(newx.T).dot(newy)
    y_hat = A[0] + A[1]*x + A[2]*(x**2) + A[3]*(x**3)
    LSE = (y_hat - y)**2
    for i in range(len(LSE)):
        sum = sum + LSE[i]
    return A,sum


def sino(x,y):
    sum = 0
    newy = y.reshape((20,1))
    newx = x.reshape((20,1))
    newx = np.sin(newx)
    ones = np.ones(newx.shape)
    newx = np.column_stack((ones,newx))
    A = np.linalg.inv(newx.T.dot(newx)).dot(newx.T).dot(newy)
    y_hat = A[0] + A[1]*np.sin(x)
    LSE = (y_hat - y)**2
    for i in range(len(LSE)):
        sum = sum + LSE[i]
    return A,sum



def cross_validation_split(x,y,folds=5):
    dataset_test_x = list()
    dataset_train_x = list()
    dataset_x = list(x)
    dataset_y = list(y)
    dataset_test_y = list()
    dataset_train_y = list()
    fold_size = int(len(x) / folds)
    for i in range(folds):
        test_x = list()
        train_x = list(x)
        test_y = list()
        train_y = list(y)
        while len(test_x) < fold_size:
            index = randrange(len(dataset_x))
            test_x.append(dataset_x.pop(index))
            train_x.pop(index)
            test_y.append(dataset_y.pop(index))
            train_y.pop(index)
        dataset_test_x.append(test_x)
        dataset_train_x.append(train_x)
        dataset_test_y.append(test_y)
        dataset_train_y.append(train_y)

    return dataset_test_x,dataset_train_x,dataset_test_y,dataset_train_y

def polydeg1_cross(test_x,train_x,test_y,train_y):
    coe = []
    LSE = []

    for newx,newy in list(zip(train_x,train_y)):
        x = np.array(newx)
        y = np.array(newy)
        y = y.reshape((16,1))
        x = x.reshape((16,1))
        ones = np.ones(x.shape)
        x = np.column_stack((ones,x))
        A = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        coe.append(A)

    for coe,testx,testy in list(zip(coe,test_x,test_y)):
        y_predict = coe[0] + coe[1]*testx
        for i in range(0,len(testy)):
            value = (y_predict[i] - testy[i])**2
            LSE.append(value)
    average = sum(LSE)/4
    return average



def polydeg3_cross(test_x,train_x,test_y,train_y):
    coe = []
    LSE = []

    for newx,newy in list(zip(train_x,train_y)):
        x = np.array(newx)
        y = np.array(newy)
        y = y.reshape((16,1))
        x = x.reshape((16,1))
        x2 = np.power(x,2)
        x3 = np.power(x,3)
        ones = np.ones(x.shape)
        x = np.column_stack((ones,x,x2,x3))
        A = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        coe.append(A)

    for coe,testx,testy in list(zip(coe,test_x,test_y)):
        y_predict = coe[0] + coe[1]*testx + coe[2]*(np.power(testx,2)) + coe[3]*(np.power(testx,3))
        for i in range(0,len(testy)):
            value = (y_predict[i] - testy[i])**2
            LSE.append(value)
    average = sum(LSE)/4
    return average



def sino_cross(test_x,train_x,test_y,train_y):
    LSE = []
    coe = []

    for newx,newy in list(zip(train_x,train_y)):
        x = np.array(newx)
        y = np.array(newy)
        y = y.reshape((16,1))
        x = x.reshape((16,1))
        newx = np.sin(x)
        ones = np.ones(x.shape)
        x = np.column_stack((ones,newx))
        A = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
        coe.append(A)


    for coe,testx,testy in list(zip(coe,test_x,test_y)):

        y_predict = coe[0] + coe[1] * np.sin(testx)
        for i in range(0,len(testy)):
            value = (y_predict[i] - testy[i])**2
            LSE.append(value)

    average = sum(LSE)/4
    return average




# test train/test split

filename = sys.argv[1]
filename = ''.join(filename)
x,y = load_points_from_file(filename)
len_data_x = len(x)

num_segments = len_data_x // 20
split_x = np.array_split(x,num_segments)
split_y = np.array_split(y,num_segments)
seed(1)
error_min_index = []
for n in range(num_segments):
    test_x,train_x,test_y,train_y = cross_validation_split(split_x[n],split_y[n],5)
    error = []



    polydeg_cross = [polydeg1_cross(test_x,train_x,test_y,train_y),polydeg3_cross(test_x,train_x,test_y,train_y),sino_cross(test_x,train_x,test_y,train_y)]

    for i in polydeg_cross:
        error.append(i)

    error_min = min(error)
    error_min_index.append(error.index(error_min))



total_error = 0
for i,o in list(zip(range(num_segments),error_min_index)):
    polydeg_plot = [polydeg1(split_x[i],split_y[i]),polydeg3(split_x[i],split_y[i]),sino(split_x[i],split_y[i])]
    A = np.array([])
    A,error = polydeg_plot[o]
    total_error = total_error + error

print('total reconstrution error:',total_error)

if len(sys.argv) >= 3 :
    fig,ax = plt.subplots()
    for i,o in list(zip(range(num_segments),error_min_index)):
        polydeg_plot = [polydeg1(split_x[i],split_y[i]),polydeg3(split_x[i],split_y[i]),sino(split_x[i],split_y[i])]
        A = np.array([])
        A,error = polydeg_plot[o]
        ax.scatter(split_x[i],split_y[i])
        A.resize(5,refcheck=False)
        if o == 2:
            x_plot = np.linspace(min(split_x[i]),max(split_x[i]),1000)
            y_plot = A[0] + A[1] * np.sin(x_plot)
            ax.plot(x_plot,y_plot)
        else:
            x_plot = np.linspace(min(split_x[i]),max(split_x[i]),1000)
            y_plot = A[0] + A[1]*x_plot + A[2]*(x_plot**2) + A[3]*(x_plot**3) + A[4]*(x_plot**4)
            ax.plot(x_plot,y_plot)


plt.show()
