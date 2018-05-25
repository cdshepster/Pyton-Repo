#!/usr/bin/python

import argparse
import cv2H
import sys
import os
import re
import math
import numpy as np
import cPickle as pickle

########################
# module: image_retrieval.py
# Cole Shepherd
# A01895001
########################

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--imgpath', required = True, help = 'image path')
ap.add_argument('-bgr', '--bgr', required = True, help = 'bgr index file to unpickle')
ap.add_argument('-hsv', '--hsv', required = True, help = 'hsv index file to unpickle')
ap.add_argument('-gsl', '--gsl', required = True, help = 'gsl index file to unpickle')
args = vars(ap.parse_args())

def mean(v):
  return sum(v)/(len(v)*1.0)

# compute the bgr similarity between
# two bgr index vectors
def bgr_img_sim(img_index_vec1, img_index_vec2):
    b,g,r = cv2.split(img_index_vec1)
    pb, pg, pr = cv2.split(img_index_vec2)

    b0List = []
    for x in xrange(0, b.shape[0]):
        b0List.append(b[x][0] * pb[x][0])
    b0 = sum(b0List)
    b1List = []
    for x in xrange(0, b.shape[0]):
        b1List.append(b[x][0] ** 2)
    b1 = sum(b1List)
    b2List = []
    for x in xrange(0, b.shape[0]):
        b2List.append(pb[x][0] ** 2)
    b2 = sum(b2List)
    b_sim = (b0) / (math.sqrt(b1) * math.sqrt(b2))

    g0List = []
    for x in xrange(0, g.shape[0]):
        g0List.append(g[x][0] * pg[x][0])
    g0 = sum(g0List)
    g1List = []
    for x in xrange(0, g.shape[0]):
        g1List.append(g[x][0] ** 2)
    g1 = sum(g1List)
    g2List = []
    for x in xrange(0, g.shape[0]):
        g2List.append(pg[x][0] ** 2)
    g2 = sum(g2List)
    g_sim = (g0) / (math.sqrt(g1) * math.sqrt(g2))

    r0List = []
    for x in xrange(0, r.shape[0]):
        r0List.append(r[x][0] * pr[x][0])
    r0 = sum(r0List)
    r1List = []
    for x in xrange(0, r.shape[0]):
        r1List.append(r[x][0] ** 2)
    r1 = sum(r1List)
    r2List = []
    for x in xrange(0, r.shape[0]):
        r2List.append(pr[x][0] ** 2)
    r2 = sum(r2List)
    r_sim = (r0) / (math.sqrt(r1) * math.sqrt(r2))

    return (b_sim+g_sim+r_sim)/3

  
# compute the hsv similarity between
# two hsv index vectors
def hsv_img_sim(img_index_vec1, img_index_vec2):
    h, s, v = cv2.split(img_index_vec1)
    ph, ps, pv = cv2.split(img_index_vec2)

    h0List = []
    for x in xrange(0, h.shape[0]):
        h0List.append(h[x][0] * ph[x][0])
    h0 = sum(h0List)
    h1List = []
    for x in xrange(0, h.shape[0]):
        h1List.append(h[x][0] ** 2)
    h1 = sum(h1List)
    h2List = []
    for x in xrange(0, h.shape[0]):
        h2List.append(ph[x][0] ** 2)
    h2 = sum(h2List)
    h_sim = (h0) / (math.sqrt(h1) * math.sqrt(h2))

    s0List = []
    for x in xrange(0, s.shape[0]):
        s0List.append(s[x][0] * ps[x][0])
    s0 = sum(s0List)
    s1List = []
    for x in xrange(0, s.shape[0]):
        s1List.append(s[x][0] ** 2)
    s1 = sum(s1List)
    s2List = []
    for x in xrange(0, s.shape[0]):
        s2List.append(ps[x][0] ** 2)
    s2 = sum(s2List)
    s_sim = (s0) / (math.sqrt(s1) * math.sqrt(s2))

    v0List = []
    for x in xrange(0, v.shape[0]):
        v0List.append(v[x][0] * pv[x][0])
    v0 = sum(v0List)
    v1List = []
    for x in xrange(0, v.shape[0]):
        v1List.append(v[x][0] ** 2)
    v1 = sum(v1List)
    v2List = []
    for x in xrange(0, v.shape[0]):
        v2List.append(pv[x][0] ** 2)
    v2 = sum(v2List)
    v_sim = (v0) / (math.sqrt(v1) * math.sqrt(v2))

    return (h_sim + s_sim + v_sim) / 3

# compute the hsv similarity between
# two gsl index vectors
def gsl_img_sim(img_index1, img_index2):
    g = img_index1
    pg = img_index2

    g0List = []
    for x in xrange(0, g.shape[0]):
        g0List.append(g[x] * pg[x])
    g0 = sum(g0List)
    g1List = []
    for x in xrange(0, g.shape[0]):
        g1List.append(g[x] ** 2)
    g1 = sum(g1List)
    g2List = []
    for x in xrange(0, g.shape[0]):
        g2List.append(pg[x] ** 2)
    g2 = sum(g2List)
    g_sim = (g0) / (math.sqrt(g1) * math.sqrt(g2))

    return g_sim

# index the input image
def index_img(imgp):
    try:
        img = cv2.imread(imgp)
        if img is None:
          print('cannot read ' + imgp)
          return
        rslt = (index_bgr(img), index_hsv(img), index_gsl(img))
        del img
        return rslt
    except Exception, e:
        print(str(e))

# this is very similar to index_bgr in image_index.py except
# you do not have to save the index in BGR_INDEX. This index
# is used to match the indices in the unpickeld BGR_INDEX.
def index_bgr(img):
    B, G, R = cv2.split(img)
    # bgri = [((b0 + b1 + b2) / 3, (g0 + g1 + g2) / 3, (r0 + r1 + r2) / 3),
    #         ((b3 + b4 + b5) / 3, (g3 + g4 + g5) / 3, (r3 + r4 + r5) / 3)],

    bAvg = []
    gAvg = []
    rAvg = []
    for line in B:
        bAvg.append(sum(line) / B.shape[1])
    for line in G:
        gAvg.append(sum(line) / G.shape[1])
    for line in R:
        rAvg.append(sum(line) / R.shape[1])

    # indexList.append((bAvg, gAvg, rAvg))
    bNew = np.asarray(bAvg, float)
    gNew = np.asarray(gAvg, float)
    rNew = np.asarray(rAvg, float)
    bgri = cv2.merge((bNew, gNew, rNew))
    return bgri

# this is very similar to index_hsv in image_index.py except
# you do not have to save the index in HSV_INDEX. This index
# is used to match the indices in the unpickeld HSV_INDEX.
def index_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
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
    return hsvi

# this is very similar to index_gs. in image_index.py except
# you do not have to save the index in GSL_INDEX. This index
# is used to match the indices in the unpickeld GSL_INDEX.
def index_gsl(img):
    gsl = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gslAvg = []
    for line in gsl:
        gslAvg.append(sum(line) / gsl.shape[1])
    gsli = np.asarray(gslAvg, float)
    return gsli

# we will unpickle into these global vars below.
BGR_INDEX = None
HSV_INDEX = None
GSL_INDEX = None

# compute the similarities between the bgr
# index vector and all the vectors in the unpickled
# bgr index bgr_index and return the top one.
def find_biggest_matches(dict, topn):
    l = []
    for key, value in sorted(dict.iteritems(), key=lambda (k, v): (v, k)):
        l.append((key, value))

    top = []
    for x in range(len(l)-topn, len(l)):
        top.append(l[x])
    return top

def compute_bgr_sim(bgr, bgr_index, topn=1):
    scoreDictionary = {}
    for item in bgr_index:
        score = bgr_img_sim(bgr, bgr_index.get(item))
        scoreDictionary[item] = score

    return find_biggest_matches(scoreDictionary, topn)

# compute the similarities between the hsv
# index vector and all the vectors in the unpickled
# hsv index hsv_index and return the top one.
def compute_hsv_sim(hsv, hsv_index, topn=1):
    scoreDictionary = {}
    for item in hsv_index:
        score = hsv_img_sim(hsv, hsv_index.get(item))
        scoreDictionary[item] = score

    return find_biggest_matches(scoreDictionary, topn)

# compute the similarities between the gsl
# index vector and all the vectors in the unpickled
# gsl index gls_index and return the top one.
def compute_gsl_sim(gsl, gsl_index, topn=1):
    scoreDictionary = {}
    for item in gsl_index:
        score = gsl_img_sim(gsl, gsl_index.get(item))
        scoreDictionary[item] = score

    return find_biggest_matches(scoreDictionary, topn)
# unpickle, match, and display
if __name__ == '__main__':
  with open(args['bgr'], 'rb') as bgrfile:
    BGR_INDEX = pickle.load(bgrfile)
  with open(args['hsv'], 'rb') as hsvfile:
    HSV_INDEX = pickle.load(hsvfile)
  with open(args['gsl'], 'rb') as gslfile:
    GSL_INDEX = pickle.load(gslfile)

  bgr, hsv, gsl = index_img(args['imgpath'])
  bgr_matches = compute_bgr_sim(bgr, BGR_INDEX)
  hsv_matches = compute_hsv_sim(hsv, HSV_INDEX)
  gsl_matches = compute_gsl_sim(gsl, GSL_INDEX)

  print bgr_matches
  print hsv_matches
  print gsl_matches

  orig = cv2.imread(args['imgpath'])
  bgr = cv2.imread(bgr_matches[0][0])
  hsv = cv2.imread(hsv_matches[0][0])
  gsl = cv2.imread(hsv_matches[0][0])
  cv2.imshow('Input', orig)
  cv2.imshow('BGR', bgr)
  cv2.imshow('HSV', hsv)
  cv2.imshow('GSL', gsl)
  cv2.waitKey()
  del orig
  del bgr
  del hsv
  del gsl
  cv2.destroyAllWindows()
    

