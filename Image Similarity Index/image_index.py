#!/usr/bin/python

import argparse
import cv2
import sys
import os
import re
import numpy as np
import cPickle as pickle

########################
# module: image_index.py
# Cole Shepherd
# a01895001
########################

ap = argparse.ArgumentParser()
ap.add_argument('-imgdir', '--imgdir', required = True, help = 'image directory')
ap.add_argument('-bgr', '--bgr', required = True, help = 'bgr index file to pickle')
ap.add_argument('-hsv', '--hsv', required = True, help = 'hsv index file to pickle')
ap.add_argument('-gsl', '--gsl', required = True, help = 'gsl index file to pickle')
args = vars(ap.parse_args())

def generate_file_names(fnpat, rootdir):
    arr = os.listdir(rootdir)
    files = []
    for x in arr:
        files.append('./'+rootdir+x)
    return files


## three index dictionaries
HSV_INDEX = {}
BGR_INDEX = {}
GSL_INDEX = {}

def index_img(imgp):
    try:
        img = cv2.imread(imgp)
        index_bgr(imgp, img)
        index_hsv(imgp, img)
        index_gsl(imgp, img)
        del img
    except Exception, e:
        print(str(e))

# compute the bgr vector for img saved in path imgp and
# index it in BGR_INDEX under imgp.
def index_bgr(imgp, img):
    B,G,R = cv2.split(img)
    # bgri = [((b0 + b1 + b2) / 3, (g0 + g1 + g2) / 3, (r0 + r1 + r2) / 3),
    #         ((b3 + b4 + b5) / 3, (g3 + g4 + g5) / 3, (r3 + r4 + r5) / 3)],

    bAvg = []
    gAvg = []
    rAvg = []
    for line in B:
        bAvg.append(sum(line)/B.shape[1])
    for line in G:
        gAvg.append(sum(line) / G.shape[1])
    for line in R:
        rAvg.append(sum(line) / R.shape[1])

    # indexList.append((bAvg, gAvg, rAvg))
    bNew = np.asarray(bAvg, float)
    gNew = np.asarray(gAvg, float)
    rNew = np.asarray(rAvg, float)
    bgri = cv2.merge((bNew, gNew, rNew))
    BGR_INDEX[imgp] = bgri



# compute the hsv vector for img saved in path imgp and
# index it in HSV_INDEX under imgp.
def index_hsv(imgp, img):
    hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    # bgri = [((b0 + b1 + b2) / 3, (g0 + g1 + g2) / 3, (r0 + r1 + r2) / 3),
    #         ((b3 + b4 + b5) / 3, (g3 + g4 + g5) / 3, (r3 + r4 + r5) / 3)],

    hAvg = []
    sAvg = []
    vAvg = []
    for line in H:
        hAvg.append(sum(line) / H.shape[1])
    for line in S:
        sAvg.append(sum(line) / S.shape[1])
    for line in V:
        vAvg.append(sum(line) / V.shape[1])

    # indexList.append((bAvg, gAvg, rAvg))
    hNew = np.asarray(hAvg, float)
    sNew = np.asarray(sAvg, float)
    vNew = np.asarray(vAvg, float)
    hsvi = cv2.merge((hNew, sNew, vNew))
    HSV_INDEX[imgp] = hsvi
# compute the gsl vector for img saved in path imgp and
# index it in GSL_INDEX under imgp.
def index_gsl(imgp, img):
    gsl = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gslAvg = []
    for line in gsl:
        gslAvg.append(sum(line) / gsl.shape[1])
    gsli = np.asarray(gslAvg, float)
    GSL_INDEX[imgp] = gsli

# index image directory imgdir
def index_img_dir(imgdir):
  print(imgdir)
  for imgp in generate_file_names(r'.+\.(jpg|png|JPG)', imgdir):
    print('indexing ' + imgp)
    index_img(imgp)
    print(imgp + ' indexed')

# index and pickle
if __name__ == '__main__':
  index_img_dir(args['imgdir'])
  with open(args['bgr'], 'wb') as bgrfile:
    pickle.dump(BGR_INDEX, bgrfile)
  with open(args['hsv'], 'wb') as hsvfile:
    pickle.dump(HSV_INDEX, hsvfile)
  with open(args['gsl'], 'wb') as gslfile:
    pickle.dump(GSL_INDEX, gslfile)
  print('indexing finished')
    

