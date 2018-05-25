#!/usr/bin/python

#######################################################
# module: img_ann_data.py
# Cole Shepherd
# a01895001
########################################################

import cv2
import numpy as np
import os

# change these values accordingly.
img_black_dir = './img_black/'
img_white_dir = './img_white/'
img_eval_black_dir = './img_eval_black/'
img_eval_white_dir = './img_eval_white/'

## training and evaluation data
DATA = []
X, y = [], []
EVAL_DATA = []
EX, ey = [], []

def normalize_image(fp):
    list = []
    finalList = []
    image = cv2.imread(fp,0)
    list = image.flatten() #at this point the list = array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n    0, 0, 0], dtype=uint8)
    for x in list:
        finalList.append(float(x/255)) #finalList = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] = 25 zeros
    return np.asarray(finalList) #again it has the \n in the debug console.

def create_data(img_dir, data_label):
    arr = os.listdir(img_dir)

    for item in arr:
        if item[0] != '.':
            path = img_dir + item
            DATA.append((path, normalize_image(path), data_label))

def create_eval_data(img_dir, data_label):
    arr = os.listdir(img_dir)

    for item in arr:
        if item[0] != '.':
            path = img_dir + item
            EVAL_DATA.append((path, normalize_image(path), data_label))

def create_Xy(DATA):
    global X, y
    for fp, img, yhat in DATA:
        X.append(img)
        y.append(yhat)
    X = np.array(X)
    y = np.array(y)

def create_EXey(EVAL_DATA):
    global EX, ey
    for fp, img, yhat in EVAL_DATA:
        EX.append(img)
        ey.append(yhat)
    EX = np.array(EX)
    ey = np.array(ey)
    
if __name__ == '__main__':
    create_data(img_black_dir, np.array([0, 1]))
    create_data(img_white_dir, np.array([1, 0]))
    np.random.shuffle(DATA)
    create_Xy(DATA)
    create_eval_data(img_eval_black_dir, np.array([0, 1]))
    create_eval_data(img_eval_white_dir, np.array([1, 0]))
    np.random.shuffle(EVAL_DATA)
    create_EXey(EVAL_DATA)











